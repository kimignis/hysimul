from typing import List, Dict

from simulation.events import BreakdownEvent
from simulation.randomizer import Randomizer
from simulation.tools import (
    get_interval,
    get_distribution,
    date_time_parse,
    ConstantDistribution,
)

r = Randomizer()

machine_classes = {}


def alt(d, a1, a2):
    return d[a1] if a1 in d else (d[a2] if a2 is not None else None)


def default(d, a1, de):
    return d[a1] if a1 in d else de


def none_is_0(v):
    return 0 if v is None else v


class Machine:

    def __init__(self, idx, d, speed):
        self.idx = idx
        self.load_time = none_is_0(get_interval(d["LTIME"], d["LTUNITS"]))
        self.unload_time = none_is_0(get_interval(d["ULTIME"], d["ULTUNITS"]))
        self.group = d["STNGRP"]
        if self.group not in machine_classes:
            machine_classes[self.group] = len(machine_classes)
        self.machine_class = machine_classes[self.group]
        self.loc = d["STNFAMLOC"]
        self.family = d["STNFAM"]
        self.cascading = (
            True
            if type(d["STNCAP"]) in [int, float] and int(d["STNCAP"]) == 2
            else False
        )
        self.speed = speed
        self.minimize_setup_time = d["WAKERESRANK"] == "wake_LeastSetupTime"

        self.available_from = None
        self.available_to = None

        self.piece_per_maintenance = []
        self.pieces_until_maintenance = []
        self.maintenance_time = []

        self.waiting_lots: List[Lot] = []
        self.next_machines: List[Machine] = []

        self.utilized_time = 0
        self.setuped_time = 0
        self.pmed_time = 0
        self.bred_time = 0
        self.starvation_time = 0
        self.starvation_count = 0
        self.blockage_time = 0
        self.blockage_count = 0

        # 시간 추적을 위한 변수들
        self.current_job = {
            "start_time": 0,  # 현재 작업 시작 시간
            "total_time": 0,  # 현재 작업의 총 소요 시간
            "remaining_time": 0,  # 남은 처리 시간
            "type": None,  # 작업 유형 (processing, setup, pm, breakdown)
            "lots_count": 0,  # 동시 처리 중인 lot 수
        }
        self.last_update_time = 0  # 마지막 업데이트 시간

        self.BLOCKAGE_THRESHOLD = 30

        self.current_setup = ""

        self.events = []
        self.min_runs_left = None
        self.min_runs_setup = None

        # RL state space features
        self.pms: List[BreakdownEvent] = []
        self.last_actions = 4 * [999]

        self.last_setup_time = 0
        self.dispatch_failed = 0

        self.has_min_runs = False

        # Machine state tracking
        self.machine_state = "idle"
        self.last_state_change = 0
        self.is_starved = False
        self.was_starved = False
        self.is_blocked = False
        self.was_blocked = False

        self.next_preventive_maintenance = None

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f"Machine {self.idx}"

    def update_state(self, current_time, new_state):
        """머신의 상태를 업데이트하고 시간을 기록합니다."""
        elapsed_time = current_time - self.last_state_change

        if self.is_starved:
            self.starvation_time += elapsed_time

        if self.is_blocked:
            self.blockage_time += elapsed_time

        if self.machine_state != new_state:
            self.machine_state = new_state
            self.last_state_change = current_time

            self.was_starved = self.is_starved
            self.is_starved = new_state == "idle" and len(self.waiting_lots) == 0

            if self.is_starved and not self.was_starved:
                self.starvation_count += 1

            self.was_blocked = self.is_blocked
            self.is_blocked = self.check_blockage()
            if self.is_blocked and not self.was_blocked:
                self.blockage_count += 1

    def check_blockage(self):
        """다음 공정 머신들의 waiting_lot 상태를 체크하여 blockage 여부를 반환합니다."""
        # 다음 머신이 없으면 blockage 아님
        if not self.next_machines:
            return False

        # processing 중이 아니면 blockage 아님
        if self.machine_state != "processing":
            return False

        # 현재 처리 중인 lot이 없으면 blockage 아님
        if not self.waiting_lots:
            return False

        # 다음 머신들의 상태 체크
        available_next_machines = [
            m for m in self.next_machines if m.machine_state not in ["breakdown", "pm"]
        ]

        # 사용 가능한 다음 머신이 없으면 blockage 아님 (이는 다른 상태로 처리되어야 함)
        if not available_next_machines:
            return False

        # 사용 가능한 다음 머신들 중 하나라도 waiting_lots가 threshold 미만이면 blockage 아님
        for next_machine in available_next_machines:
            if len(next_machine.waiting_lots) < self.BLOCKAGE_THRESHOLD:
                return False

        # 모든 사용 가능한 다음 머신의 waiting_lots가 threshold 이상이면 blockage
        return True

    def get_current_state(self):
        """현재 머신의 상태를 반환합니다."""
        return self.machine_state

    def get_starvation_time(self):
        """누적된 starvation 시간을 반환합니다."""
        return self.starvation_time

    def get_starvation_count(self):
        """총 starvation 발생 횟수를 반환합니다."""
        return self.starvation_count

    def get_blockage_time(self):
        """누적된 blockage 시간을 반환합니다."""
        return self.blockage_time

    def get_blockage_count(self):
        """총 blockage 발생 횟수를 반환합니다."""
        return self.blockage_count


class Product:
    def __init__(self, route, priority):
        self.route = route
        self.priority = priority


class Step:

    def __init__(self, idx, pieces_per_lot, d):
        self.idx = idx
        self.order = d["STEP"]
        self.step_name = d["DESC"]
        self.family = d["STNFAM"]
        self.setup_needed = d["SETUP"]
        self.setup_time = (
            get_interval(d["STIME"], d["STUNITS"]) if type(d["STIME"]) is int else None
        )
        self.rework_step = d["RWKSTEP"]
        assert len(self.family) > 0
        self.cascading = False
        if type(d["PartInterval"]) in [float, int]:
            assert d["PTPER"] == "per_piece"
            per_piece = get_interval(d["PartInterval"], d["PartIntUnits"])
            self.processing_time = get_distribution(
                d["PDIST"], d["PTUNITS"], d["PTIME"], d["PTIME2"]
            )
            self.processing_time.m += per_piece * (pieces_per_lot - 1)
            self.cascading_time = ConstantDistribution(per_piece * pieces_per_lot)
            self.cascading = True
        else:
            if d["PTPER"] == "per_piece":
                m = pieces_per_lot
            else:
                m = 1
            self.processing_time = get_distribution(
                default(d, "PDIST", "constant"),
                d["PTUNITS"],
                d["PTIME"],
                d["PTIME2"],
                multiplier=m,
            )
            if type(d["BatchInterval"]) in [float, int]:
                self.cascading_time = get_distribution(
                    "constant", d["BatchIntUnits"], d["BatchInterval"]
                )
                self.cascading = True
            else:
                self.cascading_time = self.processing_time
        self.batching = d["PTPER"] == "per_batch"
        self.batch_min = 1 if d["BATCHMN"] == "" else int(d["BATCHMN"] / pieces_per_lot)
        self.batch_max = 1 if d["BATCHMX"] == "" else int(d["BATCHMX"] / pieces_per_lot)
        self.sampling_percent = (
            100 if d["StepPercent"] in ["", None] else float(d["StepPercent"])
        )
        self.rework_percent = 0 if d["REWORK"] in ["", None] else float(d["REWORK"])

        self.cqt_for_step = d["STEP_CQT"] if "STEP_CQT" in d else None
        self.cqt_time = (
            get_interval(d["CQT"], d["CQTUNITS"])
            if self.cqt_for_step is not None
            else None
        )

        self.lot_to_lens_dedication = d["FORSTEP"] if d["SVESTN"] == "yes" else None

        self.family_location = ""
        self.transport_time = ConstantDistribution(0)

        self.reworked = {}

    def has_to_perform(self):
        if self.sampling_percent == 100:
            return True
        return r.random.uniform(0, 100) <= self.sampling_percent

    def has_to_rework(self, lot_id):
        if self.rework_percent == 0 or lot_id in self.reworked:
            return False
        self.reworked[lot_id] = True
        return r.random.uniform(0, 100) <= self.rework_percent


class Lot:
    def __init__(self, idx, route, priority, release, relative_deadline, d):
        self.idx = idx
        self.remaining_steps = [step for step in route.steps]
        self.actual_step: Step = None
        self.processed_steps = []
        self.priority = priority
        self.release_at = release
        self.deadline_at = self.release_at + relative_deadline
        self.name: str = d["LOT"]
        self.part_name: str = d["PART"]
        if "Init_" in self.name:
            self.name = self.name[self.name.index("_") + 1 : self.name.rindex("_")]

        if "CURSTEP" in d:
            cs = d["CURSTEP"]
            self.processed_steps, self.remaining_steps = (
                self.remaining_steps[: cs - 1],
                self.remaining_steps[cs - 1 :],
            )

        self.pieces = d["PIECES"]

        self.waiting_machines = []

        self.done_at = None
        self.free_since = None

        self.remaining_steps_last = -1
        self.remaining_time_last = 0

        self.dedications = {}

        self.waiting_time = 0
        self.waiting_time_batching = 0
        self.processing_time = 0
        self.transport_time = 0

        self.cqt_waiting = None
        self.cqt_deadline = None

        self.ft = None

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f"Lot {self.idx}"

    def cr(self, time):
        rt = self.remaining_time
        return (self.deadline_at - time) / rt if rt > 0 else 1

    @property
    def full_time(self):
        if self.ft is None:
            self.ft = sum(
                [
                    s.processing_time.avg()
                    for s in self.processed_steps
                    + ([self.actual_step] if self.actual_step is not None else [])
                    + self.remaining_steps
                ]
            )
        return self.ft

    @property
    def remaining_time(self):
        if self.remaining_steps_last != len(self.remaining_steps):
            if self.remaining_steps_last - 1 == len(self.remaining_steps):
                self.remaining_time_last -= self.processed_steps[
                    -1
                ].processing_time.avg()
            else:
                self.remaining_time_last = (
                    sum([s.processing_time.avg() for s in self.remaining_steps])
                    + self.actual_step.processing_time.avg()
                )
            self.remaining_steps_last = len(self.remaining_steps)
        return self.remaining_time_last


class Route:

    def __init__(self, idx, steps: List[Step]):
        self.idx = idx
        self.steps = steps


class FileRoute(Route):

    def __init__(self, idx, pieces_per_lot, steps: List[Dict]):
        steps = [Step(i, pieces_per_lot, d) for i, d in enumerate(steps)]
        super().__init__(idx, steps)
