import sys; args = sys.argv[1:]
from pathlib import Path
import argparse
import csv
import math

root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path: sys.path.insert(0, str(root))

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


def setstyle():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.facecolor": "#F7F8FA",
        "axes.facecolor": "#F7F8FA",
        "axes.edgecolor": "#A0AEC0",
        "axes.labelcolor": COLORS["dark"],
        "axes.titlecolor": COLORS["dark"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "grid.alpha": 0.25,
        "savefig.facecolor": "#F7F8FA",
        "savefig.bbox": "tight",
    })


def newcube():
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


def blankenergy():
    return {key: 0.0 for key in ["conduction", "convection", "longwave", "solar", "radiation", "internal", "total"]}


def addrow(rows, suite, scenario, t, obj, energy, details):
    row = {
        "suite": suite,
        "scenario": scenario,
        "time_s": round(float(t), 6),
        "time_min": round(float(t) / 60.0, 6),
        "temperature_K": obj.temperature,
        "ambient_K": AMBIENT,
    }
    row.update(details)
    row.update({f"q_{key}_W": obj.last_rates[key] for key in obj.last_rates})
    row.update({f"energy_{key}_kJ": energy[key] / 1000.0 for key in energy})
    rows.append(row)


def runseries(suite, scenario, duration, dt, details=None, **conditions):
    obj, rows, energy = newcube(), [], blankenergy()
    details = {} if details is None else details
    addrow(rows, suite, scenario, 0.0, obj, energy, details)
    for index in range(1, int(round(duration / dt)) + 1):
        obj.step(dt, **conditions)
        for key, watts in obj.last_rates.items():
            energy[key] += watts * dt
        addrow(rows, suite, scenario, index * dt, obj, energy, details)
    return rows


def savecsv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, path):
    fig.savefig(path, dpi=190)
    plt.close(fig)


def showpath(path):
    try:
        return path.relative_to(root)
    except ValueError:
        return str(path).encode("ascii", "backslashreplace").decode("ascii")


def conduction(duration, dt, datadir, graphdir):
    cases = [("hot_surface", HOT_SURFACE, COLORS["hot"]), ("cold_surface", COLD_SURFACE, COLORS["cold"])]
    allrows = []
    for name, surface, linecolor in cases:
        rows = runseries(
            "conduction",
            name,
            duration,
            dt,
            details={"surface_K": surface},
            contact_temp=surface,
        )
        allrows += rows
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot([r["time_min"] for r in rows], [r["temperature_K"] for r in rows], color=linecolor, linewidth=2.6, label="Cube temperature")
        ax.axhline(surface, color=linecolor, linestyle="--", alpha=0.65, label=f"Surface ({surface:.0f} K)")
        ax.axhline(START_TEMP, color=COLORS["muted"], linestyle=":", linewidth=1.5, label=f"Initial cube ({START_TEMP:.0f} K)")
        direction = "hot" if surface > START_TEMP else "cold"
        fig.suptitle(f"Conduction from a {direction} surface", x=0.12, y=0.98, ha="left", fontsize=16, fontweight="bold", color=COLORS["dark"])
        fig.text(0.12, 0.91, f"Fourier conduction only | 0.25 m2 contact | k = 15 W/(m K) | {duration / 3600.0:g} h", color=COLORS["muted"])
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Cube temperature (K)")
        ax.legend(loc="best")
        fig.subplots_adjust(top=0.84)
        savefig(fig, graphdir / f"0{1 if surface > START_TEMP else 2}_conduction_{direction}.png")
    savecsv(allrows, datadir / "conduction.csv")
    return allrows


def convection(interval, dt, datadir, graphdir):
    rows, summary = [], []
    for wind in np.arange(0.0, 30.0 + 1e-9, 1.0):
        case = runseries(
            "convection",
            f"wind_{wind:.0f}",
            interval,
            dt,
            details={"wind_m_s": wind},
            ambient_temp=AMBIENT,
            wind_speed=wind,
        )
        rows += case
        final = case[-1]
        summary.append({
            "wind_m_s": wind,
            "interval_s": interval,
            "heat_removed_kJ": -final["energy_convection_kJ"],
            "final_temperature_K": final["temperature_K"],
            "temperature_change_K": final["temperature_K"] - START_TEMP,
        })

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    winds = [r["wind_m_s"] for r in summary]
    removed = [r["heat_removed_kJ"] for r in summary]
    finals = [r["final_temperature_K"] for r in summary]
    axes[0].plot(winds, removed, color=COLORS["cold"], linewidth=2.4)
    axes[0].scatter(winds, removed, color=COLORS["cold"], s=22, zorder=3)
    axes[0].set_ylabel("Heat removed (kJ)")
    axes[1].plot(winds, finals, color=COLORS["shade"], linewidth=2.4)
    axes[1].scatter(winds, finals, color=COLORS["shade"], s=22, zorder=3)
    axes[1].axhline(AMBIENT, color=COLORS["muted"], linestyle=":", label="Ambient temperature")
    axes[1].set_xlabel("Wind speed (m/s)")
    axes[1].set_ylabel("Final cube temperature (K)")
    axes[1].legend(loc="best")
    sample = newcube()
    transition = 5e5 * sample.air_nu / max(sample.dimensions)
    for ax in axes:
        ax.axvline(transition, color=COLORS["muted"], linestyle="--", linewidth=1.2, alpha=0.75)
    axes[0].text(transition + 0.4, min(removed) + 35, "approx. flow transition", rotation=90, color=COLORS["muted"], fontsize=9)
    fig.suptitle("Convective heat removal vs. wind speed", x=0.12, y=0.985, ha="left", fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.12, 0.94, f"Constant {interval / 60.0:g} min interval | ambient {AMBIENT:.0f} K | initial cube {START_TEMP:.0f} K", color=COLORS["muted"])
    fig.subplots_adjust(top=0.87, hspace=0.25)
    savefig(fig, graphdir / "03_convection_wind_sweep.png")
    savecsv(rows, datadir / "convection_timeseries.csv")
    savecsv(summary, datadir / "convection_summary.csv")
    return rows, summary


def radiation(duration, dt, datadir, graphdir):
    cases = [("sun", 1.0, COLORS["sun"]), ("shade", SHADE_FRACTION, COLORS["shade"])]
    allrows = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    for name, fraction, linecolor in cases:
        rows = runseries(
            "radiation",
            name,
            duration,
            dt,
            details={"sun_fraction": fraction, "solar_W_m2": SUN_W_M2},
            surroundings_temp=AMBIENT,
            solar_irradiance=SUN_W_M2,
            sun_fraction=fraction,
            sun_direction=(0.0, 0.0, 1.0),
        )
        allrows += rows
        label = "In sun" if name == "sun" else "In shade"
        axes[0].plot([r["time_min"] for r in rows], [r["energy_radiation_kJ"] for r in rows], color=linecolor, linewidth=2.5, label=label)
        axes[1].plot([r["time_min"] for r in rows], [r["temperature_K"] for r in rows], color=linecolor, linewidth=2.5, label=label)
    axes[0].axhline(0, color=COLORS["muted"], linewidth=1)
    axes[0].set_ylabel("Cumulative net heat (kJ)")
    axes[0].legend(loc="best")
    axes[1].axhline(START_TEMP, color=COLORS["muted"], linestyle=":", linewidth=1.5)
    axes[1].set_xlabel("Elapsed time (min)")
    axes[1].set_ylabel("Cube temperature (K)")
    axes[1].legend(loc="best")
    fig.suptitle("Net radiative heat: sun vs. shade", x=0.13, y=0.985, ha="left", fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.13, 0.94, f"Solar plus long-wave radiation | {SUN_W_M2:.0f} W/m2 sun | {SHADE_FRACTION:.0%} diffuse shade | surroundings {AMBIENT:.0f} K", color=COLORS["muted"])
    fig.subplots_adjust(top=0.87, hspace=0.25)
    savefig(fig, graphdir / "04_radiation_sun_vs_shade.png")
    savecsv(allrows, datadir / "radiation.csv")
    return allrows


def combined(duration, dt, datadir, graphdir):
    contacts = [("hot", HOT_SURFACE), ("cold", COLD_SURFACE)]
    light = [("sun", 1.0), ("shade", SHADE_FRACTION)]
    winds = [0.0, 10.0, 20.0, 30.0]
    windcolors = plt.cm.viridis(np.linspace(0.12, 0.88, len(winds)))
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for rowindex, (contactname, surface) in enumerate(contacts):
        for colindex, (lightname, fraction) in enumerate(light):
            ax = axes[rowindex, colindex]
            for wind, linecolor in zip(winds, windcolors):
                scenario = f"{contactname}_{lightname}_wind_{wind:.0f}"
                case = runseries(
                    "combined",
                    scenario,
                    duration,
                    dt,
                    details={
                        "surface_type": contactname,
                        "surface_K": surface,
                        "light": lightname,
                        "sun_fraction": fraction,
                        "wind_m_s": wind,
                    },
                    ambient_temp=AMBIENT,
                    surroundings_temp=AMBIENT,
                    wind_speed=wind,
                    solar_irradiance=SUN_W_M2,
                    sun_fraction=fraction,
                    sun_direction=(0.0, 0.0, 1.0),
                    contact_temp=surface,
                )
                rows += case
                ax.plot([r["time_min"] for r in case], [r["temperature_K"] for r in case], color=linecolor, linewidth=2.0, label=f"{wind:.0f} m/s")
            ax.axhline(START_TEMP, color=COLORS["muted"], linestyle=":", linewidth=1.2)
            ax.set_title(f"{contactname.title()} surface | {lightname.title()}", loc="left", fontsize=12)

    fig.suptitle("Combined conduction, convection, and radiation", x=0.08, y=0.98, ha="left", fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.text(0.08, 0.935, f"Hot {HOT_SURFACE:.0f} K | cold {COLD_SURFACE:.0f} K | sun {SUN_W_M2:.0f} W/m2 | shade {SHADE_FRACTION:.0%} | ambient {AMBIENT:.0f} K", color=COLORS["muted"])
    fig.supxlabel("Elapsed time (min)")
    fig.supylabel("Cube temperature (K)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.94, 0.98), ncol=2, title="Wind speed")
    fig.subplots_adjust(top=0.86, left=0.09, right=0.96, bottom=0.09, hspace=0.27, wspace=0.15)
    savefig(fig, graphdir / "05_combined_effects_matrix.png")
    savecsv(rows, datadir / "combined_matrix.csv")
    return rows


def sanitychecks(conductionrows, convsummary, radiationrows, combinedrows, path):
    final = lambda rows, name: [r for r in rows if r["scenario"] == name][-1]
    hotrow = final(conductionrows, "hot_surface")
    coldrow = final(conductionrows, "cold_surface")
    hot = hotrow["temperature_K"]
    cold = coldrow["temperature_K"]
    sun = final(radiationrows, "sun")
    shade = final(radiationrows, "shade")
    removed = [r["heat_removed_kJ"] for r in convsummary]
    scenarios = {r["scenario"] for r in combinedrows}
    cube = newcube()
    conductance = cube.k * cube.contact_area / cube.contact_length
    tau = cube.thermal_mass / conductance
    hotexact = HOT_SURFACE + (START_TEMP - HOT_SURFACE) * math.exp(-hotrow["time_s"] / tau)
    coldexact = COLD_SURFACE + (START_TEMP - COLD_SURFACE) * math.exp(-coldrow["time_s"] / tau)
    conduction_error = max(abs(hot - hotexact), abs(cold - coldexact))
    convection_error = 0.0
    exposed = cube.surface_area - cube.contact_area
    for row in convsummary:
        rate = cube.convectioncoefficient(row["wind_m_s"]) * exposed / cube.thermal_mass
        exact = AMBIENT + (START_TEMP - AMBIENT) * math.exp(-rate * row["interval_s"])
        convection_error = max(convection_error, abs(row["final_temperature_K"] - exact))
    lastrows = {}
    for row in radiationrows + combinedrows:
        lastrows[row["scenario"]] = row
    energy_error = max(abs((row["temperature_K"] - START_TEMP) * cube.thermal_mass / 1000.0 - row["energy_total_kJ"]) for row in lastrows.values())
    checks = [
        {"check": "hot contact warms cube", "passed": hot > START_TEMP, "value": hot, "expected": f"> {START_TEMP}"},
        {"check": "cold contact cools cube", "passed": cold < START_TEMP, "value": cold, "expected": f"< {START_TEMP}"},
        {"check": "conduction matches analytical solution", "passed": conduction_error < 0.05, "value": conduction_error, "expected": "< 0.05 K max error"},
        {"check": "convection removal rises with wind", "passed": all(removed[i] <= removed[i + 1] + 1e-9 for i in range(len(removed) - 1)), "value": removed[-1] - removed[0], "expected": "> 0 kJ spread"},
        {"check": "convection matches analytical solution", "passed": convection_error < 0.05, "value": convection_error, "expected": "< 0.05 K max error"},
        {"check": "sun adds more radiation than shade", "passed": sun["energy_radiation_kJ"] > shade["energy_radiation_kJ"], "value": sun["energy_radiation_kJ"] - shade["energy_radiation_kJ"], "expected": "> 0 kJ spread"},
        {"check": "sun case ends warmer than shade", "passed": sun["temperature_K"] > shade["temperature_K"], "value": sun["temperature_K"] - shade["temperature_K"], "expected": "> 0 K spread"},
        {"check": "integrated heat matches temperature change", "passed": energy_error < 1e-6, "value": energy_error, "expected": "< 1e-6 kJ max error"},
        {"check": "combined matrix has all cases", "passed": len(scenarios) == 16, "value": len(scenarios), "expected": "16 scenarios"},
        {"check": "combined values are finite", "passed": all(math.isfinite(r["temperature_K"]) for r in combinedrows), "value": len(combinedrows), "expected": "all rows finite"},
    ]
    savecsv(checks, path)
    return checks


def main():
    parser = argparse.ArgumentParser(description="Run isolated and combined thermal sanity checks")
    parser.add_argument("--output", default=str(root / "logs" / "thermal_sanity"))
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=5.0)
    options = parser.parse_args(args)
    if options.duration_hours <= 0 or options.interval_minutes <= 0 or options.dt <= 0:
        parser.error("durations and dt must be positive")

    outdir = Path(options.output)
    if not outdir.is_absolute(): outdir = root / outdir
    datadir, graphdir = outdir / "data", outdir / "graphs"
    datadir.mkdir(parents=True, exist_ok=True)
    graphdir.mkdir(parents=True, exist_ok=True)
    setstyle()

    duration = options.duration_hours * 3600.0
    interval = options.interval_minutes * 60.0
    conductionrows = conduction(duration, options.dt, datadir, graphdir)
    _, convsummary = convection(interval, options.dt, datadir, graphdir)
    radiationrows = radiation(duration, options.dt, datadir, graphdir)
    combinedrows = combined(duration, options.dt, datadir, graphdir)
    checks = sanitychecks(conductionrows, convsummary, radiationrows, combinedrows, datadir / "sanity_checks.csv")

    passed = sum(bool(check["passed"]) for check in checks)
    print(f"thermal sanity checks: {passed}/{len(checks)} passed")
    print(f"data: {showpath(datadir)}")
    print(f"graphs: {showpath(graphdir)}")
    if passed != len(checks): raise SystemExit(1)


if __name__ == "__main__": main()

# Jerry Huang, 1, 2027
