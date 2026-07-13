import sys; args = sys.argv[1:]
from pathlib import Path
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt

# runs a day-night sweep and saves the raw data, stats, and graphs
# usage: thermal_sanity_check.py [output folder] [duration sec] [step sec]
# this checks model trends, not whether the coefficients match real data
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path: sys.path.insert(0, str(root))

from sim.Environment.Thermal.thermal_manager import ThermalManager


# heat rates are K/s; shade and air are saved for context
terms = ["sun", "conv", "rad", "contact", "internal", "shade", "air"]


# fresh baseline each time so runs do not share changed values
def basecase():
    return {
        "mass": 4.0, "area": 1.0, "volume": 0.20, "cp": 850.0,
        "absorpt": 0.65, "conductivity": 15.0, "contact_area": 0.02,
        "heat_watts": 0.0, "wind": 0.0, "irr": 1.0,
        "ambient": 293.0, "sky": 260.0, "contact": None
    }


# same robot-like material used by the manager tests
def material(case):
    return {
        "alpha": 3e-3, "beta": 1e-3, "gamma": 9e-11, "emiss": 0.80,
        "T": 285.0, "cp": case["cp"], "density": 1200.0,
        "conductivity": case["conductivity"], "absorpt": case["absorpt"]
    }


# only change one value per case
def makecase(var, val):
    case = basecase()
    if var != "baseline": case[var] = val
    return case


# clear day: sunrise at 6, noon peak, sunset at 18, hottest air at 15
def dayenv(case, t):
    hour = (t/3600.0) % 24.0
    sun = max(0.0, np.sin(np.pi*(hour - 6.0)/12.0))
    return {"hour": hour, "irradiance": case["irr"]*sun, "ambient_K": case["ambient"] + 7.0*np.sin(2*np.pi*(hour - 9.0)/24.0), "sky_K": case["sky"] + 10.0*sun}


# flatten the current state for the main csv
def addrow(rows, scen, var, val, t, obj, env):
    row = {"scenario": scen, "variable": var, "value": "" if val is None else val, "time_s": round(t, 6), "time_h": round(t/3600.0, 6), "hour": env["hour"], "irradiance": env["irradiance"], "ambient_K": env["ambient_K"], "sky_K": env["sky_K"], "temp_K": obj.T, "temp_C": obj.T - 273.15}
    row.update({k: obj.last_terms.get(k, 0.0) for k in terms})
    rows.append(row)


# run one level with its own manager and object
def runone(var, val, duration, dt):
    case, rows = makecase(var, val), []
    label = "baseline" if var == "baseline" else f"{var}={val}"
    # fake body id is enough since geometry is passed in
    tm = ThermalManager(time_of_day=12, ambient_K=case["ambient"], T_sky=case["sky"])
    tm.add_object(1, material=material(case), area=case["area"], volume=case["volume"], mass=case["mass"], cp=case["cp"], absorpt=case["absorpt"], conductivity=case["conductivity"], contact_area=case["contact_area"], heat_watts=case["heat_watts"])
    obj = tm.objects[(1, -1)]

    # fake contact lets this run without a pybullet world
    if case["contact"] is not None:
        target = float(case["contact"])
        obj.contact_term = lambda temps=None, obj=obj, target=target: obj.k*obj.contact_area*(target - obj.T)/obj.therm_mass

    # hide one warmup day so startup does not skew mass and cp
    warmup = 86400.0 if duration >= 86400.0 else 0.0
    for i in range(1, int(warmup/dt) + 1):
        env = dayenv(case, i*dt)
        tm.time_of_day, tm.ambient, tm.T_sky = env["hour"], env["ambient_K"], env["sky_K"]
        tm.update(dt, env["irradiance"], wind=case["wind"])

    # save midnight, then one row each step
    env = dayenv(case, 0.0)
    addrow(rows, label, var, val, 0.0, obj, env)
    for i in range(1, int(duration/dt) + 1):
        t = i*dt
        env = dayenv(case, t)
        tm.time_of_day, tm.ambient, tm.T_sky = env["hour"], env["ambient_K"], env["sky_K"]
        tm.update(dt, env["irradiance"], wind=case["wind"])
        addrow(rows, label, var, val, t, obj, env)
    return rows


# save every timestep
def savecsv(rows, path):
    keys = ["scenario", "variable", "value", "time_s", "time_h", "hour", "irradiance", "ambient_K", "sky_K", "temp_K", "temp_C"] + terms
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# average tied ranks for spearman
def rank(vals):
    out = []
    for val in vals:
        spots = [i + 1 for i, x in enumerate(sorted(vals)) if x == val]
        out.append(float(np.mean(spots)))
    return out


# exact spearman works better here since time points are not independent
# p shows ordered sensitivity, not whether the model matches real measurements
def spearman(vals, response):
    x, y = rank(vals), rank(response)
    if np.std(x) == 0 or np.std(y) == 0: return 0.0, 1.0
    rho = float(np.corrcoef(x, y)[0, 1])
    possible = list(itertools.permutations(y))
    extreme = sum(abs(float(np.corrcoef(x, p)[0, 1])) >= abs(rho) - 1e-12 for p in possible)
    return rho, extreme/len(possible)


# check if the levels move one way
def trend(vals, finals):
    pairs = sorted(zip(vals, finals), key=lambda x: -1 if x[0] is None else float(x[0]))
    ys = [p[1] for p in pairs]
    if all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1)): return "up"
    if all(ys[i] >= ys[i + 1] - 1e-9 for i in range(len(ys) - 1)): return "down"
    return "mixed"


# summarize each sweep and compare its direction to the expected one
# mass, area, and cp use daily swing since they mainly change inertia
def savesensitivity(rows, sweeps, path):
    expected = {"mass": "down", "area": "up", "absorpt": "up", "wind": "down", "heat_watts": "up", "irr": "up", "ambient": "up", "sky": "up", "cp": "down", "contact": "up"}
    # use baseline swing to put the effect on a simple relative scale
    base = [r["temp_K"] for r in rows if r["scenario"] == "baseline"]
    baseamp = float(np.ptp(base))
    out = []
    for var, vals in sweeps.items():
        # skip the shared midnight point in summaries
        scen = [f"{var}={v}" for v in vals]
        groups = [[r for r in rows if r["scenario"] == s and r["time_s"] > 0] for s in scen]
        means = [float(np.mean([r["temp_K"] for r in g])) for g in groups]
        peaks = [max(r["temp_K"] for r in g) for g in groups]
        lows = [min(r["temp_K"] for r in g) for g in groups]
        swings = [peaks[i] - lows[i] for i in range(len(vals))]
        metric = "daily_swing_K" if var in ["mass", "area", "cp"] else "daily_mean_K"
        response = swings if metric == "daily_swing_K" else means
        rho, p = spearman([float(v) for v in vals], response)
        got = trend(vals, response)
        # widest gap between levels at the same time
        curves = np.asarray([[r["temp_K"] for r in g] for g in groups])
        spread = float(np.max(np.ptp(curves, axis=0)))
        out.append({"variable": var, "levels": "; ".join(map(str, vals)), "response_metric": metric, "response_values": "; ".join(f"{x:.4f}" for x in response), "daily_means_K": "; ".join(f"{x:.4f}" for x in means), "daily_swings_K": "; ".join(f"{x:.4f}" for x in swings), "daily_mean_spread_K": max(means) - min(means), "peak_spread_K": max(peaks) - min(peaks), "low_spread_K": max(lows) - min(lows), "max_curve_spread_K": spread, "effect_vs_daily_swing": spread/baseamp if baseamp > 0 else "", "meaningful_over_0.05K": spread >= 0.05, "trend": got, "expected": expected[var], "direction_ok": got == expected[var], "spearman_rho": rho, "exact_p": p, "note": "exact rank test on one cycle metric per run; effect sizes and direction matter most for this deterministic model"})
    with open(path, "w", newline="") as f:
        keys = ["variable", "levels", "response_metric", "response_values", "daily_means_K", "daily_swings_K", "daily_mean_spread_K", "peak_spread_K", "low_spread_K", "max_curve_spread_K", "effect_vs_daily_swing", "meaningful_over_0.05K", "trend", "expected", "direction_ok", "spearman_rho", "exact_p", "note"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(out)


# one graph per variable plus the baseline
def plottemps(rows, outdir, variables):
    for var in variables:
        fig, ax = plt.subplots(figsize=(9, 5))
        scenarios = sorted({r["scenario"] for r in rows if r["variable"] in [var, "baseline"]})
        for scen in scenarios:
            pts = [r for r in rows if r["scenario"] == scen]
            ax.plot([r["time_h"] for r in pts], [r["temp_K"] for r in pts], label=scen)
        ax.set_title(f"temperature response: {var}")
        ax.set_xlabel("time since midnight (hours)")
        ax.set_ylabel("temperature (K)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"temperature_{var}.png", dpi=160)
        plt.close(fig)


# positive terms warm the baseline and negative ones cool it
def plotterms(rows, outdir):
    pts = [r for r in rows if r["scenario"] == "baseline"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in ["sun", "conv", "rad", "contact", "internal"]:
        ax.plot([r["time_h"] for r in pts], [r[k] for r in pts], label=k)
    ax.set_title("baseline heat terms")
    ax.set_xlabel("time since midnight (hours)")
    ax.set_ylabel("temperature rate (K/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "terms_baseline.png", dpi=160)
    plt.close(fig)


# show what drove the baseline over the day
def plotenv(rows, outdir):
    pts = [r for r in rows if r["scenario"] == "baseline"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r["time_h"] for r in pts], [r["ambient_K"] for r in pts], label="ambient K")
    ax.plot([r["time_h"] for r in pts], [r["sky_K"] for r in pts], label="sky K")
    ax.set_xlabel("time since midnight (hours)")
    ax.set_ylabel("temperature (K)")
    sunax = ax.twinx()
    sunax.plot([r["time_h"] for r in pts], [r["irradiance"] for r in pts], color="goldenrod", label="sun")
    sunax.set_ylabel("normalized irradiance")
    ax.set_title("day-night environment cycle")
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + sunax.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "environment_cycle.png", dpi=160)
    plt.close(fig)


# keep printed paths short when possible
def showpath(path):
    try: return path.relative_to(root)
    except ValueError: return path


# default is one full day at one minute per step
def main():
    outdir = Path(args[0]) if len(args) > 0 else root / "logs" / "thermal_sanity"
    if not outdir.is_absolute(): outdir = root / outdir
    duration = float(args[1]) if len(args) > 1 else 86400.0
    dt = float(args[2]) if len(args) > 2 else 60.0
    outdir.mkdir(parents=True, exist_ok=True)
    graphdir = outdir / "graphs"
    graphdir.mkdir(parents=True, exist_ok=True)

    # five levels gives the rank test enough to work with
    sweeps = {
        "mass": [1.0, 2.0, 4.0, 8.0, 12.0],
        "area": [0.4, 0.7, 1.0, 1.8, 3.0],
        "absorpt": [0.25, 0.45, 0.65, 0.8, 0.95],
        "wind": [0.0, 2.0, 4.0, 7.0, 10.0],
        "heat_watts": [0.0, 4.0, 8.0, 16.0, 25.0],
        "irr": [0.0, 0.25, 0.5, 0.75, 1.0],
        "ambient": [283.0, 288.0, 293.0, 298.0, 303.0],
        "sky": [240.0, 250.0, 260.0, 270.0, 280.0],
        "cp": [500.0, 650.0, 850.0, 1200.0, 1600.0],
        "contact": [275.0, 285.0, 293.0, 303.0, 315.0],
    }

    rows = runone("baseline", "base", duration, dt)
    for var, vals in sweeps.items():
        for val in vals: rows += runone(var, val, duration, dt)

    savecsv(rows, outdir / "thermal_sanity.csv")
    savesensitivity(rows, sweeps, outdir / "thermal_sensitivity.csv")
    plottemps(rows, graphdir, list(sweeps.keys()))
    plotterms(rows, graphdir)
    plotenv(rows, graphdir)
    print(f"wrote {showpath(outdir / 'thermal_sanity.csv')}")
    print(f"wrote {showpath(outdir / 'thermal_sensitivity.csv')}")
    print(f"wrote graphs to {showpath(graphdir)}")


if __name__ == "__main__": main()