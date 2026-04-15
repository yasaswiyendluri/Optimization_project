import os, sys, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from src.models    import MachineModel, QualityModel, MaintenanceModel, fit_weibull
from src.optimizer import asa_pso, vanilla_pso

os.makedirs("results", exist_ok=True)


def load():
    if not os.path.exists(cfg.DATA_PATH):
        print("Dataset not found. Place ai4i2020.csv inside data/")
        sys.exit(1)

    df = pd.read_csv(cfg.DATA_PATH)
    df["stage"] = df[cfg.TYPE_COL].astype(str).str.strip().str[0].str.upper()

    df["TTF"] = 0.0
    for stage in cfg.STAGES:
        rows  = df.index[df["stage"] == stage].tolist()
        count = 0.0
        for i in rows:
            count += 1.0
            df.at[i, "TTF"] = count
            if df.at[i, cfg.FAILURE_COL] == 1:
                count = 0.0

    print(f"Loaded {len(df)} rows | Failures: {int(df[cfg.FAILURE_COL].sum())}")
    return df


def run_eda(df):
    print("\nEDA Summary")
    for s in cfg.STAGES:
        sub = df[df["stage"] == s]
        print(f"Stage {s}: rows={len(sub)}, failure_rate={round(sub[cfg.FAILURE_COL].mean()*100,2)}%, avg_wear={round(sub[cfg.WEAR_COL].mean(),1)}")

    counts = [int(df[c].sum()) for c in cfg.FAILURE_MODE_COLS]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cfg.FAILURE_MODE_COLS, counts)
    ax.set_title("Failure Mode Counts")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/eda_failure_modes.png", dpi=130)
    plt.close()


def run_models(df):
    print("\nModel Results")

    machines = []
    for s in cfg.STAGES:
        sub       = df[(df["stage"] == s) & (df[cfg.FAILURE_COL] == 1)]
        beta, eta = fit_weibull(sub["TTF"].tolist())
        m         = MachineModel(s, beta, eta)
        machines.append(m)

        print(f"Stage {s}: beta={round(beta,3)}, eta={round(eta,2)}, R_next={round(m.survival(cfg.NEXT_TASK_HOURS),4)}")

    times, probs = machines[0].state_probs_over_time(cfg.TASK_HOURS * 5)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j in range(len(probs)):
        ax.plot(times, probs[j], linewidth=1.8, label=MachineModel.LABELS[j])
    ax.axvline(cfg.TASK_HOURS, linestyle="--", alpha=0.5)
    ax.set_title("State Probabilities")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/state_probs.png", dpi=130)
    plt.close()

    qm = QualityModel.from_data(df)
    print("\nQuality Model")
    for s in cfg.STAGES:
        print(f"Stage {s}: nu={round(qm.nu_rates[s],7)}, Q0={round(qm.Q(s,0),3)}, Q_end={round(qm.Q(s,cfg.TASK_HOURS),4)}")

    t_vals = list(range(0, cfg.TASK_HOURS * 5 + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in cfg.STAGES:
        ax.plot(t_vals, [qm.Q(s, t) for t in t_vals], linewidth=1.8, label=f"Stage {s}")
    ax.axvline(cfg.TASK_HOURS, linestyle="--", alpha=0.5)
    ax.set_title("WIP Quality")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/quality_curves.png", dpi=130)
    plt.close()

    flows = qm.min_input(cfg.TASK_HOURS)
    print("\nMin Input Flow:")
    for s, f in flows.items():
        print(f"Stage {s}: {round(f,2)}")

    return machines, qm


def run_optimize(machines):
    print("\nOptimization")

    mm = MaintenanceModel(machines)

    best_rsys, best_g = -1, None
    for seed in range(5):
        random.seed(seed)
        pos, _, _ = asa_pso(mm.objective, len(cfg.STAGES),
                            n=cfg.N_PARTICLES, iters=cfg.MAX_ITER)
        g = mm.decode(pos)
        if mm.feasible(g) and mm.R_sys(g) > best_rsys:
            best_rsys = mm.R_sys(g)
            best_g    = g[:]

    if best_g is None:
        best_g = [4, 4, 4]
    g = best_g

    print(f"Optimal policy: {g}")
    print(f"Cost={round(mm.total_cost(g),1)}, Time={round(mm.total_time(g),2)}, R_sys={round(mm.R_sys(g),4)}, Feasible={mm.feasible(g)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(cfg.STAGES)))
    ax.bar(x, [mm.cost(i, g[i]) for i in x], label="Cost", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} (g={g[i]})" for i, s in enumerate(cfg.STAGES)])
    ax.set_title("Maintenance Plan")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/maintenance_result.png", dpi=130)
    plt.close()

    return mm


def run_compare(machines):
    print("\nConvergence Comparison")

    n_iter = 150
    mm = MaintenanceModel(machines)

    asa_runs, pso_runs = [], []
    for seed in range(10):
        random.seed(seed)
        _, _, c = asa_pso(mm.objective, len(cfg.STAGES),
                          n=cfg.N_PARTICLES, iters=n_iter)
        asa_runs.append(c)

        random.seed(seed)
        _, _, c = vanilla_pso(mm.objective, len(cfg.STAGES),
                              n=cfg.N_PARTICLES, iters=n_iter)
        pso_runs.append(c)

    asa_avg = [sum(r[k] for r in asa_runs) / 10 for k in range(n_iter)]
    pso_avg = [sum(r[k] for r in pso_runs) / 10 for k in range(n_iter)]

    print(f"ASA-PSO final: {round(asa_avg[-1],5)}")
    print(f"PSO final: {round(pso_avg[-1],5)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, n_iter + 1), pso_avg, linestyle="--", label="PSO")
    ax.plot(range(1, n_iter + 1), asa_avg, label="ASA-PSO")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/convergence.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("Selective Maintenance Optimization")

    df           = load()
    run_eda(df)
    machines, qm = run_models(df)
    run_optimize(machines)
    run_compare(machines)

    print("\nDone. Results saved in /results")