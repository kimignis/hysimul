import datetime
import io
import json
import statistics
from collections import defaultdict
import os

from simulation.classes import Lot, Step


def print_statistics(instance, days, dataset, disp, method="greedy", dir="greedy"):
    from simulation.instance import Instance

    instance: Instance
    lot: Lot

    # 시뮬레이션별 폴더 생성
    sim_folder = f"{dir}/{method}_{days}days_{dataset}_{disp}"
    if not os.path.exists(sim_folder):
        os.makedirs(sim_folder)

    # 현재 시간이 1시간 단위인지 확인 (3600초 = 1시간)
    current_hour = int(instance.current_time / 3600)

    # 통계 수집 및 출력 함수
    def collect_and_print_stats():
        # current_time이 0일 때는 통계를 출력하지 않음
        if instance.current_time == 0:
            return

        lots = defaultdict(
            lambda: {
                "ACT": [],
                "throughput": 0,
                "on_time": 0,
                "tardiness": 0,
                "waiting_time": 0,
                "processing_time": 0,
                "transport_time": 0,
                "waiting_time_batching": 0,
            }
        )
        apt = {}
        dl = {}

        for lot in instance.done_lots:
            lots[lot.name]["ACT"].append(lot.done_at - lot.release_at)
            lots[lot.name]["throughput"] += 1
            lots[lot.name]["tardiness"] += max(0, lot.done_at - lot.deadline_at)
            lots[lot.name]["waiting_time"] += lot.waiting_time
            lots[lot.name]["waiting_time_batching"] += lot.waiting_time_batching
            lots[lot.name]["processing_time"] += lot.processing_time
            lots[lot.name]["transport_time"] += lot.transport_time
            if lot.done_at <= lot.deadline_at:
                lots[lot.name]["on_time"] += 1
            if lot.name not in apt:
                apt[lot.name] = sum(
                    [s.processing_time.avg() for s in lot.processed_steps]
                )
                dl[lot.name] = lot.deadline_at - lot.release_at

        # Lot 통계 출력
        print("\n=== Lot Statistics at Hour", current_hour, "===")
        print("Lot", "APT", "DL", "ACT", "TH", "ONTIME", "tardiness", "wa", "pr", "tr")
        acts = []
        ths = []
        ontimes = []
        for lot_name in sorted(list(lots.keys())):
            l = lots[lot_name]
            avg = statistics.mean(l["ACT"]) / 3600 / 24
            lots[lot_name]["ACT"] = avg
            acts += [avg]
            th = lots[lot_name]["throughput"]
            ths += [th]
            ontime = round(l["on_time"] / l["throughput"] * 100)
            ontimes += [ontime]
            wa = lots[lot_name]["waiting_time"] / l["throughput"] / 3600 / 24
            wab = lots[lot_name]["waiting_time_batching"] / l["throughput"] / 3600 / 24
            pr = lots[lot_name]["processing_time"] / l["throughput"] / 3600 / 24
            tr = lots[lot_name]["transport_time"] / l["throughput"] / 3600 / 24
            print(
                lot_name,
                round(apt[lot_name] / 3600 / 24, 1),
                round(dl[lot_name] / 3600 / 24, 1),
                round(avg, 1),
                th,
                ontime,
                l["tardiness"],
                wa,
                wab,
                pr,
                tr,
            )

        if acts:  # acts 리스트가 비어있지 않은 경우에만 통계 출력
            print("---------------")
            print(
                round(statistics.mean(acts), 2),
                statistics.mean(ths),
                statistics.mean(ontimes),
            )
            print(round(sum(acts), 2), sum(ths), sum(ontimes))
        else:
            print("---------------")
            print("No completed lots yet")

        # 머신 통계 수집
        utilized_times = defaultdict(lambda: [])
        setup_times = defaultdict(lambda: [])
        pm_times = defaultdict(lambda: [])
        br_times = defaultdict(lambda: [])
        starvation_times = defaultdict(lambda: [])
        starvation_counts = defaultdict(lambda: [])
        blockage_times = defaultdict(lambda: [])
        blockage_counts = defaultdict(lambda: [])

        for machine in instance.machines:
            utilized_times[machine.family].append(machine.utilized_time)
            setup_times[machine.family].append(machine.setuped_time)
            pm_times[machine.family].append(machine.pmed_time)
            br_times[machine.family].append(machine.bred_time)
            starvation_times[machine.family].append(machine.starvation_time)
            starvation_counts[machine.family].append(machine.starvation_count)
            blockage_times[machine.family].append(machine.blockage_time)
            blockage_counts[machine.family].append(machine.blockage_count)

        # 통계 출력
        print(f"\n=== Statistics at Hour {current_hour} ===")
        print(
            "Machine",
            "Cnt",
            "avail",
            "util",
            "br",
            "pm",
            "setup",
            "starv",
            "starv_cnt",
            "block",
            "block_cnt",
            "waiting_lots",
        )

        machines = defaultdict(lambda: {})
        for machine_name in sorted(list(utilized_times.keys())):
            av = (
                instance.current_time
                - statistics.mean(pm_times[machine_name])
                - statistics.mean(br_times[machine_name])
            )
            machines[machine_name]["avail"] = av / instance.current_time

            # family 계산 제거하고 각 머신의 평균값 사용
            machines[machine_name]["util"] = (
                statistics.mean(utilized_times[machine_name]) / av
            )
            machines[machine_name]["pm"] = (
                statistics.mean(pm_times[machine_name]) / instance.current_time
            )
            machines[machine_name]["br"] = (
                statistics.mean(br_times[machine_name]) / instance.current_time
            )
            machines[machine_name]["setup"] = (
                statistics.mean(setup_times[machine_name]) / instance.current_time
            )
            machines[machine_name]["starv"] = (
                statistics.mean(starvation_times[machine_name]) / instance.current_time
            )
            machines[machine_name]["starv_cnt"] = round(
                statistics.mean(starvation_counts[machine_name]), 1
            )
            machines[machine_name]["block"] = (
                statistics.mean(blockage_times[machine_name]) / instance.current_time
            )
            machines[machine_name]["block_cnt"] = round(
                statistics.mean(blockage_counts[machine_name]), 1
            )

            # waiting_lots 정보 추가
            waiting_lots_count = sum(
                len(m.waiting_lots) for m in instance.family_machines[machine_name]
            )
            machines[machine_name]["waiting_lots"] = waiting_lots_count

            print(
                machine_name,
                len(utilized_times[machine_name]),
                round(machines[machine_name]["avail"] * 100, 2),
                round(machines[machine_name]["util"] * 100, 2),
                round(machines[machine_name]["br"] * 100, 2),
                round(machines[machine_name]["pm"] * 100, 2),
                round(machines[machine_name]["setup"] * 100, 2),
                round(machines[machine_name]["starv"] * 100, 2),
                machines[machine_name]["starv_cnt"],
                round(machines[machine_name]["block"] * 100, 2),
                machines[machine_name]["block_cnt"],
                machines[machine_name]["waiting_lots"],
            )

        # JSON 파일로 저장
        output_data = {
            "hour": current_hour,
            "lots": lots,
            "machines": machines,
            "plugins": {
                plugin.get_output_name(): plugin.get_output_value()
                for plugin in instance.plugins
                if plugin.get_output_name() is not None
            },
        }

        with io.open(f"{sim_folder}/hour_{current_hour}.json", "w") as f:
            json.dump(output_data, f)

    # 1시간마다 통계 수집 및 출력
    collect_and_print_stats()
