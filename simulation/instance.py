from collections import defaultdict
from typing import Dict, List, Set, Tuple

from simulation.classes import Machine, Route, Lot
from simulation.dispatching.dm_lot_for_machine import LotForMachineDispatchManager
from simulation.dispatching.dm_machine_for_lot import MachineForLotDispatchManager
from simulation.event_queue import EventQueue
from simulation.events import (
    MachineDoneEvent,
    LotDoneEvent,
    BreakdownEvent,
    ReleaseEvent,
)
from simulation.plugins.interface import IPlugin


class Instance:

    def __init__(
        self,
        machines: List[Machine],
        routes: Dict[str, Route],
        lots: List[Lot],
        setups: Dict[Tuple, int],
        setup_min_run: Dict[str, int],
        breakdowns: List[BreakdownEvent],
        lot_for_machine,
        plugins,
    ):
        self.plugins: List[IPlugin] = plugins
        self.lot_waiting_at_machine = defaultdict(lambda: (0, 0))

        self.free_machines: List[bool] = []
        self.usable_machines: Set[Machine] = set()
        self.usable_lots: List[Lot] = list()

        self.machines: List[Machine] = [m for m in machines]
        self.family_machines = defaultdict(lambda: [])
        for m in self.machines:
            self.family_machines[m.family].append(m)
        self.routes: Dict[str, Route] = routes
        self.setups: Dict[Tuple, int] = setups
        self.setup_min_run: Dict[str, int] = setup_min_run

        # 머신들의 next_machines 초기화
        self._initialize_next_machines()

        self.dm = (
            LotForMachineDispatchManager()
            if lot_for_machine
            else MachineForLotDispatchManager()
        )
        self.dm.init(self)

        self.dispatchable_lots: List[Lot] = lots
        self.dispatchable_lots.sort(key=lambda k: k.release_at)
        self.active_lots: List[Lot] = []
        self.done_lots: List[Lot] = []

        self.events = EventQueue()

        self.current_time = 0
        self.last_stats_hour = -1  # 마지막으로 통계를 출력한 시간 (시간 단위)

        for plugin in self.plugins:
            plugin.on_sim_init(self)

        self.next_step()

        self.free_up_machines(self.machines)

        for br in breakdowns:
            self.add_event(br)

        self.printed_days = -1

    @property
    def current_time_days(self):
        return self.current_time / 3600 / 24

    def next_step(self):
        process_until = []
        if len(self.events.arr) > 0:
            process_until.append(max(0, self.events.first.timestamp))
        process_until.append(max(0, self.dispatchable_lots[0].release_at))
        process_until = min(process_until)

        # 현재 시간이 1시간 단위를 넘어갔는지 확인
        current_hour = int(self.current_time / 3600)
        next_hour = int(process_until / 3600)

        # 시간이 변경되었고 진행 중인 작업이 있는 머신들의 시간 업데이트
        if next_hour > current_hour:
            for machine in self.machines:
                if machine.current_job["remaining_time"] > 0:
                    # 다음 시간대의 시작
                    next_hour_start = (current_hour + 1) * 3600

                    def accumulate_remaining_time(total_time, time_type, lots_count=1):
                        time_in_next_hour = min(
                            3600, total_time  # 한 시간  # 남은 처리 시간
                        )
                        remaining_time = total_time - time_in_next_hour

                        # 시간 타입에 따라 적절한 속성에 누적
                        if time_type == "processing":
                            machine.utilized_time += time_in_next_hour / lots_count
                        elif time_type == "setup":
                            machine.setuped_time += time_in_next_hour
                        elif time_type == "pm":
                            machine.pmed_time += time_in_next_hour
                        elif time_type == "breakdown":
                            machine.bred_time += time_in_next_hour

                        return remaining_time

                    # 남은 시간 처리
                    machine.current_job["remaining_time"] = accumulate_remaining_time(
                        machine.current_job["remaining_time"],
                        machine.current_job["type"],
                        machine.current_job["lots_count"],
                    )
                    machine.last_update_time = next_hour_start

        if current_hour > self.last_stats_hour:
            self.last_stats_hour = current_hour
            from simulation.stats import print_statistics

            if hasattr(self, "stats_params"):
                print_statistics(
                    self,
                    self.stats_params["days"],
                    self.stats_params["dataset"],
                    self.stats_params["disp"],
                    method=self.stats_params["method"],
                )

        while len(self.events.arr) > 0 and self.events.first.timestamp <= process_until:
            ev = self.events.pop_first()
            self.current_time = max(0, ev.timestamp, self.current_time)
            ev.handle(self)
        ReleaseEvent.handle(self, process_until)

    def free_up_machines(self, machines):
        # add machine to list of available machines
        for machine in machines:
            machine.events.clear()
            self.dm.free_up_machine(self, machine)
            # idle 상태로 업데이트
            machine.update_state(self.current_time, "idle")

            for plugin in self.plugins:
                plugin.on_machine_free(self, machine)

    def free_up_lots(self, lots: List[Lot]):
        # add lot to lists, make it available
        for lot in lots:
            lot.free_since = self.current_time
            step_found = False
            while len(lot.remaining_steps) > 0:
                old_step = None
                if lot.actual_step is not None:
                    lot.processed_steps.append(lot.actual_step)
                    old_step = lot.actual_step
                if lot.actual_step is not None and lot.actual_step.has_to_rework(
                    lot.idx
                ):
                    rw_step = lot.actual_step.rework_step
                    removed = lot.processed_steps[rw_step - 1 :]
                    lot.processed_steps = lot.processed_steps[: rw_step - 1]
                    lot.remaining_steps = removed + lot.remaining_steps
                lot.actual_step, lot.remaining_steps = (
                    lot.remaining_steps[0],
                    lot.remaining_steps[1:],
                )
                if lot.actual_step.has_to_perform():
                    # print(f'Lot {lot.idx} step {len(lot.processed_steps)} / {len(lot.remaining_steps)}')
                    self.dm.free_up_lots(self, lot)
                    step_found = True
                    for plugin in self.plugins:
                        plugin.on_step_done(self, lot, old_step)
                    break
            if not step_found:
                assert len(lot.remaining_steps) == 0
                lot.actual_step = None
                lot.done_at = self.current_time
                # print(f'Lot {lot.idx} is done {len(self.active_lots)} {len(self.done_lots)} {self.current_time_days}')
                self.active_lots.remove(lot)
                self.done_lots.append(lot)
                for plugin in self.plugins:
                    plugin.on_lot_done(self, lot)

            for plugin in self.plugins:
                plugin.on_lot_free(self, lot)

    def dispatch(self, machine: Machine, lots: List[Lot]):
        # remove machine and lot from active sets
        self.reserve_machine_lot(lots, machine)
        lwam = self.lot_waiting_at_machine[machine.family]
        self.lot_waiting_at_machine[machine.family] = (
            lwam[0] + len(lots),
            lwam[1] + sum([self.current_time - l.free_since for l in lots]),
        )
        for lot in lots:
            lot.waiting_time += self.current_time - lot.free_since
            if lot.actual_step.batch_max > 1:
                lot.waiting_time_batching += self.current_time - lot.free_since
            if lot.actual_step.cqt_for_step is not None:
                lot.cqt_waiting = lot.actual_step.cqt_for_step
                lot.cqt_deadline = lot.actual_step.cqt_time
            if lot.actual_step.order == lot.cqt_waiting:
                if lot.cqt_deadline < self.current_time:
                    for plugin in self.plugins:
                        plugin.on_cqt_violated(self, machine, lot)
                lot.cqt_waiting = None
                lot.cqt_deadline = None

        # compute times for lot and machine
        lot_time, machine_time, setup_time = self.get_times(self.setups, lots, machine)

        # 머신 상태 업데이트 (setup_time 계산 후로 이동)
        if setup_time > 0:
            machine.update_state(self.current_time, "setup")
        else:
            machine.update_state(self.current_time, "processing")

        # compute per-piece preventive maintenance requirement
        for i in range(len(machine.pieces_until_maintenance)):
            machine.pieces_until_maintenance[i] -= sum([l.pieces for l in lots])
            if machine.pieces_until_maintenance[i] <= 0:
                s = machine.maintenance_time[i].sample()
                machine_time += s
                machine.pieces_until_maintenance[i] = machine.piece_per_maintenance[i]
                machine.pmed_time += s
                # PM 상태로 업데이트
                machine.update_state(self.current_time + machine_time - s, "pm")
        # if there is ltl dedication, dedicate lot for selected step
        for lot in lots:
            if lot.actual_step.lot_to_lens_dedication is not None:
                lot.dedications[lot.actual_step.lot_to_lens_dedication] = machine.idx
        # decrease / eliminate min runs required before next setup
        if machine.min_runs_left is not None:
            machine.min_runs_left -= len(lots)
            if machine.min_runs_left <= 0:
                machine.min_runs_left = None
                machine.min_runs_setup = None
        # add events
        machine_done = self.current_time + machine_time + setup_time
        lot_done = self.current_time + lot_time + setup_time
        ev1 = MachineDoneEvent(machine_done, [machine])
        ev2 = LotDoneEvent(lot_done, [machine], lots)
        self.add_event(ev1)
        self.add_event(ev2)
        machine.events += [ev1, ev2]

        for plugin in self.plugins:
            plugin.on_dispatch(self, machine, lots, machine_done, lot_done)
        return machine_done, lot_done

    def get_times(self, setups, lots, machine):
        proc_t_samp = lots[0].actual_step.processing_time.sample()
        lot_time = proc_t_samp + machine.load_time + machine.unload_time
        for lot in lots:
            lot.processing_time += lot_time
        if len(lots[0].remaining_steps) > 0:
            tt = lots[0].remaining_steps[0].transport_time.sample()
            lot_time += tt
            for lot in lots:
                lot.transport_time += tt
        if lots[0].actual_step.processing_time == lots[0].actual_step.cascading_time:
            cascade_t_samp = proc_t_samp
        else:
            cascade_t_samp = lots[0].actual_step.cascading_time.sample()
        machine_time = cascade_t_samp + (
            machine.load_time + machine.unload_time if not machine.cascading else 0
        )

        new_setup = lots[0].actual_step.setup_needed
        if new_setup != "" and machine.current_setup != new_setup:
            if lots[0].actual_step.setup_time is not None:
                setup_time = lots[0].actual_step.setup_time
            elif (machine.current_setup, new_setup) in setups:
                setup_time = setups[(machine.current_setup, new_setup)]
            elif ("", new_setup) in setups:
                setup_time = setups[("", new_setup)]
            else:
                setup_time = 0
        else:
            setup_time = 0

        # 시간 분할 및 누적을 위한 함수
        def accumulate_time(total_time, time_type, start_time, lots_count=1):
            remaining_time = total_time
            current_process_time = start_time

            while remaining_time > 0:
                # 현재 시간이 속한 시간대의 끝
                current_hour_end = ((current_process_time // 3600) + 1) * 3600

                # 현재 시간대에서 처리할 수 있는 시간 계산
                time_in_current_hour = min(
                    current_hour_end - current_process_time,  # 현재 시간대의 남은 시간
                    remaining_time,  # 남은 처리 시간
                )

                # 시간 타입에 따라 적절한 속성에 누적
                if time_type == "processing":
                    machine.utilized_time += time_in_current_hour / lots_count
                elif time_type == "setup":
                    machine.setuped_time += time_in_current_hour
                elif time_type == "pm":
                    machine.pmed_time += time_in_current_hour
                elif time_type == "breakdown":
                    machine.bred_time += time_in_current_hour

                remaining_time -= time_in_current_hour
                current_process_time = current_hour_end

            return total_time - remaining_time

        # setup time 처리
        if setup_time > 0:
            machine.current_job = {
                "start_time": self.current_time,
                "total_time": setup_time,
                "remaining_time": setup_time
                - accumulate_time(setup_time, "setup", self.current_time),
                "type": "setup",
                "lots_count": 1,
            }
            machine.last_setup_time = setup_time

        # processing time 처리
        process_start_time = self.current_time + setup_time
        machine.current_job = {
            "start_time": process_start_time,
            "total_time": machine_time,
            "remaining_time": machine_time
            - accumulate_time(
                machine_time, "processing", process_start_time, len(lots)
            ),
            "type": "processing",
            "lots_count": len(lots),
        }

        machine.last_update_time = self.current_time

        if new_setup in self.setup_min_run:
            machine.min_runs_left = self.setup_min_run[new_setup]
            machine.min_runs_setup = new_setup
            machine.has_min_runs = True

        machine.last_setup = machine.current_setup
        machine.current_setup = new_setup

        return lot_time, machine_time, setup_time

    def reserve_machine_lot(self, lots, machine):
        self.dm.reserve(self, lots, machine)

    def add_event(self, to_insert):
        # insert event to the correct place in the array
        self.events.ordered_insert(to_insert)

    def next_decision_point(self):
        return self.dm.next_decision_point(self)

    def handle_breakdown(self, machine, delay):
        ta = []
        for ev in machine.events:
            if ev in self.events.arr:
                ta.append(ev)
                self.events.remove(ev)
        for ev in ta:
            ev.timestamp += delay
            self.add_event(ev)
        # breakdown 상태로 업데이트
        machine.update_state(self.current_time, "breakdown")

    @property
    def done(self):
        return len(self.dispatchable_lots) == 0 and len(self.active_lots) == 0

    def finalize(self):
        for plugin in self.plugins:
            plugin.on_sim_done(self)

    def print_progress_in_days(self):
        import sys

        if int(self.current_time_days) > self.printed_days:
            self.printed_days = int(self.current_time_days)
            if self.printed_days > 0:
                sys.stderr.write(
                    f"\rDay {self.printed_days}===Throughput: {round(len(self.done_lots) / self.printed_days)}/day="
                )
                sys.stderr.flush()

    def _initialize_next_machines(self):
        """각 머신의 next_machines 리스트를 route 정보를 기반으로 초기화합니다."""
        # 각 route의 step sequence를 분석하여 머신 간 연결 관계 파악
        machine_family_connections = defaultdict(set)  # family 간의 연결 관계

        for route in self.routes.values():
            for i in range(len(route.steps) - 1):
                current_family = route.steps[i].family
                next_family = route.steps[i + 1].family
                machine_family_connections[current_family].add(next_family)

        # 각 머신의 next_machines 설정
        for machine in self.machines:
            next_families = machine_family_connections[machine.family]
            for next_family in next_families:
                if next_family in self.family_machines:
                    machine.next_machines.extend(self.family_machines[next_family])
