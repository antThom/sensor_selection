import argparse
import csv
import math
import sys
from pathlib import Path


args = sys.argv[1:]
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.Environment.ThermalObject import ThermalObject


AMBIENT = 270.0
START_TEMP = 300.0
HOT_SURFACE = 340.0
COLD_SURFACE = 240.0
SUN_W_M2 = 1000.0
SHADE_FRACTION = 0.10

COLORS = {
    "hot": "#C44E52",
    "cold": "#4C78A8",
    "sun": "#D99A25",
    "shade": "#2A6F74",
    "dark": "#243447",
    "muted": "#718096",
}


def set_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F8FA",
            "axes.facecolor": "#F7F8FA",
            "grid.alpha": 0.25,
            "savefig.facecolor": "#F7F8FA",
            "savefig.bbox": "tight",
        }
    )


def new_cube():
    return ThermalObject(
        temperature=START_TEMP,
        dimensions=(0.5, 0.5, 0.5),
        mass=35.0,
        specific_heat=900.0,
        conductivity=15.0,
        emissivity=0.85,
        absorptivity=0.85,
        contact_area=0.25,
        contact_length=0.25,
    )


def blank_energy():
    return {
        key: 0.0
        for key in (
            "conduction",
            "convection",
            "longwave",
            "solar",
            "radiation",
            "internal",
            "total",
        )
    }


def add_row(rows, suite, scenario, elapsed, thermal_object, energy, details):
    row = {
        "suite": suite,
        "scenario": scenario,
        "time_s": round(float(elapsed), 6),
        "time_min": round(float(elapsed) / 60.0, 6),
        "temperature_K": thermal_object.temperature,
        "ambient_K": AMBIENT,
    }
    row.update(details)
    row.update(
        {
            f"q_{key}_W": thermal_object.last_rates[key]
            for key in thermal_object.last_rates
        }
    )
    row.update(
        {
            f"energy_{key}_kJ": energy[key] / 1000.0
            for key in energy
        }
    )
    rows.append(row)


def run_series(
    suite,
    scenario,
    duration,
    dt,
    details=None,
    **conditions,
):
    thermal_object = new_cube()
    rows = []
    energy = blank_energy()
    details = {} if details is None else details
    add_row(
        rows,
        suite,
        scenario,
        0.0,
        thermal_object,
        energy,
        details,
    )
    for index in range(1, int(round(duration / dt)) + 1):
        thermal_object.step(dt, **conditions)
        for key, watts in thermal_object.last_rates.items():
            energy[key] += watts * dt
        add_row(
            rows,
            suite,
            scenario,
            index * dt,
            thermal_object,
            energy,
            details,
        )
    return rows


def save_csv(rows, path):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, path):
    figure.savefig(path, dpi=190)
    plt.close(figure)


def display_path(path):
    try:
        return path.relative_to(root)
    except ValueError:
        return str(path).encode(
            "ascii",
            "backslashreplace",
        ).decode("ascii")


def run_conduction(duration, dt, data_directory, graph_directory):
    cases = (
        ("hot_surface", HOT_SURFACE, COLORS["hot"]),
        ("cold_surface", COLD_SURFACE, COLORS["cold"]),
    )
    all_rows = []
    for name, surface_temperature, line_color in cases:
        rows = run_series(
            "conduction",
            name,
            duration,
            dt,
            details={"surface_K": surface_temperature},
            contact_temp=surface_temperature,
        )
        all_rows.extend(rows)
        figure, axis = plt.subplots(figsize=(9, 5.2))
        axis.plot(
            [row["time_min"] for row in rows],
            [row["temperature_K"] for row in rows],
            color=line_color,
            linewidth=2.6,
            label="Cube temperature",
        )
        axis.axhline(
            surface_temperature,
            color=line_color,
            linestyle="--",
            alpha=0.65,
            label=f"Surface ({surface_temperature:.0f} K)",
        )
        axis.axhline(
            START_TEMP,
            color=COLORS["muted"],
            linestyle=":",
            label=f"Initial cube ({START_TEMP:.0f} K)",
        )
        axis.set_title(f"Conduction from a {name.split('_')[0]} surface")
        axis.set_xlabel("Elapsed time (min)")
        axis.set_ylabel("Cube temperature (K)")
        axis.legend(loc="best")
        graph_number = 1 if surface_temperature > START_TEMP else 2
        direction = "hot" if graph_number == 1 else "cold"
        save_figure(
            figure,
            graph_directory
            / f"0{graph_number}_conduction_{direction}.png",
        )
    save_csv(all_rows, data_directory / "conduction.csv")
    return all_rows


def run_convection(interval, dt, data_directory, graph_directory):
    rows = []
    summary = []
    for wind in np.arange(0.0, 30.0 + 1e-9, 1.0):
        case = run_series(
            "convection",
            f"wind_{wind:.0f}",
            interval,
            dt,
            details={"wind_m_s": wind},
            ambient_temp=AMBIENT,
            wind_speed=wind,
        )
        rows.extend(case)
        final = case[-1]
        summary.append(
            {
                "wind_m_s": wind,
                "interval_s": interval,
                "heat_removed_kJ": -final["energy_convection_kJ"],
                "final_temperature_K": final["temperature_K"],
                "temperature_change_K": (
                    final["temperature_K"] - START_TEMP
                ),
            }
        )

    figure, axes = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    winds = [row["wind_m_s"] for row in summary]
    removed = [row["heat_removed_kJ"] for row in summary]
    final_temperatures = [
        row["final_temperature_K"] for row in summary
    ]
    axes[0].plot(winds, removed, color=COLORS["cold"], linewidth=2.4)
    axes[0].set_ylabel("Heat removed (kJ)")
    axes[1].plot(
        winds,
        final_temperatures,
        color=COLORS["shade"],
        linewidth=2.4,
    )
    axes[1].axhline(
        AMBIENT,
        color=COLORS["muted"],
        linestyle=":",
        label="Ambient temperature",
    )
    axes[1].set_xlabel("Wind speed (m/s)")
    axes[1].set_ylabel("Final cube temperature (K)")
    axes[1].legend(loc="best")
    figure.suptitle("Convective heat removal vs. wind speed")
    save_figure(
        figure,
        graph_directory / "03_convection_wind_sweep.png",
    )
    save_csv(rows, data_directory / "convection_timeseries.csv")
    save_csv(summary, data_directory / "convection_summary.csv")
    return rows, summary


def run_radiation(duration, dt, data_directory, graph_directory):
    cases = (
        ("sun", 1.0, COLORS["sun"]),
        ("shade", SHADE_FRACTION, COLORS["shade"]),
    )
    all_rows = []
    figure, axes = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    for name, fraction, line_color in cases:
        rows = run_series(
            "radiation",
            name,
            duration,
            dt,
            details={
                "sun_fraction": fraction,
                "solar_W_m2": SUN_W_M2,
            },
            surroundings_temp=AMBIENT,
            solar_irradiance=SUN_W_M2,
            sun_fraction=fraction,
            sun_direction=(0.0, 0.0, 1.0),
        )
        all_rows.extend(rows)
        label = "In sun" if name == "sun" else "In shade"
        axes[0].plot(
            [row["time_min"] for row in rows],
            [row["energy_radiation_kJ"] for row in rows],
            color=line_color,
            linewidth=2.5,
            label=label,
        )
        axes[1].plot(
            [row["time_min"] for row in rows],
            [row["temperature_K"] for row in rows],
            color=line_color,
            linewidth=2.5,
            label=label,
        )
    axes[0].set_ylabel("Cumulative net heat (kJ)")
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Elapsed time (min)")
    axes[1].set_ylabel("Cube temperature (K)")
    axes[1].legend(loc="best")
    figure.suptitle("Net radiative heat: sun vs. shade")
    save_figure(
        figure,
        graph_directory / "04_radiation_sun_vs_shade.png",
    )
    save_csv(all_rows, data_directory / "radiation.csv")
    return all_rows


def run_combined(duration, dt, data_directory, graph_directory):
    contacts = (("hot", HOT_SURFACE), ("cold", COLD_SURFACE))
    lighting = (("sun", 1.0), ("shade", SHADE_FRACTION))
    winds = (0.0, 10.0, 20.0, 30.0)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(winds)))
    rows = []
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )

    for row_index, (contact_name, surface_temperature) in enumerate(
        contacts
    ):
        for column_index, (light_name, fraction) in enumerate(lighting):
            axis = axes[row_index, column_index]
            for wind, line_color in zip(winds, colors):
                scenario = (
                    f"{contact_name}_{light_name}_wind_{wind:.0f}"
                )
                case = run_series(
                    "combined",
                    scenario,
                    duration,
                    dt,
                    details={
                        "surface_type": contact_name,
                        "surface_K": surface_temperature,
                        "light": light_name,
                        "sun_fraction": fraction,
                        "wind_m_s": wind,
                    },
                    ambient_temp=AMBIENT,
                    surroundings_temp=AMBIENT,
                    wind_speed=wind,
                    solar_irradiance=SUN_W_M2,
                    sun_fraction=fraction,
                    sun_direction=(0.0, 0.0, 1.0),
                    contact_temp=surface_temperature,
                )
                rows.extend(case)
                axis.plot(
                    [entry["time_min"] for entry in case],
                    [entry["temperature_K"] for entry in case],
                    color=line_color,
                    linewidth=2.0,
                    label=f"{wind:.0f} m/s",
                )
            axis.set_title(
                f"{contact_name.title()} surface | {light_name.title()}"
            )

    figure.suptitle(
        "Combined conduction, convection, and radiation"
    )
    figure.supxlabel("Elapsed time (min)")
    figure.supylabel("Cube temperature (K)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=2)
    save_figure(
        figure,
        graph_directory / "05_combined_effects_matrix.png",
    )
    save_csv(rows, data_directory / "combined_matrix.csv")
    return rows


def perform_sanity_checks(
    conduction_rows,
    convection_summary,
    radiation_rows,
    combined_rows,
    path,
):
    def final(rows, scenario):
        return [row for row in rows if row["scenario"] == scenario][-1]

    hot_row = final(conduction_rows, "hot_surface")
    cold_row = final(conduction_rows, "cold_surface")
    sun_row = final(radiation_rows, "sun")
    shade_row = final(radiation_rows, "shade")

    cube = new_cube()
    conductance = cube.k * cube.contact_area / cube.contact_length
    time_constant = cube.thermal_mass / conductance
    hot_exact = HOT_SURFACE + (START_TEMP - HOT_SURFACE) * math.exp(
        -hot_row["time_s"] / time_constant
    )
    cold_exact = COLD_SURFACE + (START_TEMP - COLD_SURFACE) * math.exp(
        -cold_row["time_s"] / time_constant
    )
    conduction_error = max(
        abs(hot_row["temperature_K"] - hot_exact),
        abs(cold_row["temperature_K"] - cold_exact),
    )

    convection_error = 0.0
    exposed_area = cube.surface_area - cube.contact_area
    for row in convection_summary:
        rate = (
            cube.convection_coefficient(row["wind_m_s"])
            * exposed_area
            / cube.thermal_mass
        )
        exact = AMBIENT + (START_TEMP - AMBIENT) * math.exp(
            -rate * row["interval_s"]
        )
        convection_error = max(
            convection_error,
            abs(row["final_temperature_K"] - exact),
        )

    last_rows = {}
    for row in radiation_rows + combined_rows:
        last_rows[row["scenario"]] = row
    energy_error = max(
        abs(
            (row["temperature_K"] - START_TEMP)
            * cube.thermal_mass
            / 1000.0
            - row["energy_total_kJ"]
        )
        for row in last_rows.values()
    )

    removed = [
        row["heat_removed_kJ"] for row in convection_summary
    ]
    scenarios = {row["scenario"] for row in combined_rows}
    checks = [
        {
            "check": "hot contact warms cube",
            "passed": hot_row["temperature_K"] > START_TEMP,
            "value": hot_row["temperature_K"],
            "expected": f"> {START_TEMP}",
        },
        {
            "check": "cold contact cools cube",
            "passed": cold_row["temperature_K"] < START_TEMP,
            "value": cold_row["temperature_K"],
            "expected": f"< {START_TEMP}",
        },
        {
            "check": "conduction matches analytical solution",
            "passed": conduction_error < 0.05,
            "value": conduction_error,
            "expected": "< 0.05 K max error",
        },
        {
            "check": "convection removal rises with wind",
            "passed": all(
                removed[index] <= removed[index + 1] + 1e-9
                for index in range(len(removed) - 1)
            ),
            "value": removed[-1] - removed[0],
            "expected": "> 0 kJ spread",
        },
        {
            "check": "convection matches analytical solution",
            "passed": convection_error < 0.05,
            "value": convection_error,
            "expected": "< 0.05 K max error",
        },
        {
            "check": "sun adds more radiation than shade",
            "passed": (
                sun_row["energy_radiation_kJ"]
                > shade_row["energy_radiation_kJ"]
            ),
            "value": (
                sun_row["energy_radiation_kJ"]
                - shade_row["energy_radiation_kJ"]
            ),
            "expected": "> 0 kJ spread",
        },
        {
            "check": "sun case ends warmer than shade",
            "passed": (
                sun_row["temperature_K"] > shade_row["temperature_K"]
            ),
            "value": (
                sun_row["temperature_K"] - shade_row["temperature_K"]
            ),
            "expected": "> 0 K spread",
        },
        {
            "check": "integrated heat matches temperature change",
            "passed": energy_error < 1e-6,
            "value": energy_error,
            "expected": "< 1e-6 kJ max error",
        },
        {
            "check": "combined matrix has all cases",
            "passed": len(scenarios) == 16,
            "value": len(scenarios),
            "expected": "16 scenarios",
        },
        {
            "check": "combined values are finite",
            "passed": all(
                math.isfinite(row["temperature_K"])
                for row in combined_rows
            ),
            "value": len(combined_rows),
            "expected": "all rows finite",
        },
    ]
    save_csv(checks, path)
    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Run isolated and combined thermal sanity checks"
    )
    parser.add_argument(
        "--output",
        default=str(root / "logs" / "thermal_sanity"),
    )
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=5.0)
    options = parser.parse_args(args)
    if (
        options.duration_hours <= 0
        or options.interval_minutes <= 0
        or options.dt <= 0
    ):
        parser.error("durations and dt must be positive")

    output_directory = Path(options.output)
    if not output_directory.is_absolute():
        output_directory = root / output_directory
    data_directory = output_directory / "data"
    graph_directory = output_directory / "graphs"
    data_directory.mkdir(parents=True, exist_ok=True)
    graph_directory.mkdir(parents=True, exist_ok=True)
    set_style()

    duration = options.duration_hours * 3600.0
    interval = options.interval_minutes * 60.0
    conduction_rows = run_conduction(
        duration,
        options.dt,
        data_directory,
        graph_directory,
    )
    _, convection_summary = run_convection(
        interval,
        options.dt,
        data_directory,
        graph_directory,
    )
    radiation_rows = run_radiation(
        duration,
        options.dt,
        data_directory,
        graph_directory,
    )
    combined_rows = run_combined(
        duration,
        options.dt,
        data_directory,
        graph_directory,
    )
    checks = perform_sanity_checks(
        conduction_rows,
        convection_summary,
        radiation_rows,
        combined_rows,
        data_directory / "sanity_checks.csv",
    )

    passed = sum(bool(check["passed"]) for check in checks)
    print(f"thermal sanity checks: {passed}/{len(checks)} passed")
    print(f"data: {display_path(data_directory)}")
    print(f"graphs: {display_path(graph_directory)}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
