# src/models.py
# Core models from the paper — kept simple for coursework.
#
# MachineModel        → Weibull degradation + state probabilities (Sec 3.2)
# QualityModel        → WIP pass rate Q(t)                        (Sec 3.3)
# MaintenanceModel    → cost, time, robustness, objective          (Sec 4.2)

import math
import numpy as np
import config as cfg



# Weibull helpers  (paper Eq. 4, 6, 22)


def weibull_survival(t, beta, eta):
    """R(t) = exp(-(t/eta)^beta)   →  Paper Eq.(22)"""
    if t <= 0:
        return 1.0
    return math.exp(-((t / eta) ** beta))

def weibull_cdf(t, beta, eta):
    """F(t) = 1 - R(t)   →  Paper Eq.(6)"""
    return 1.0 - weibull_survival(t, beta, eta)

def fit_weibull(ttf_list):
    """
    Fit Weibull shape (beta) and scale (eta) from time-to-failure data.
    Uses log-std approximation — good enough for coursework.
    """
    data = [x for x in ttf_list if x > 0]
    if len(data) < 2:
        return 1.0, 50.0          # fallback defaults
    ln_x = np.log(data)
    beta = float(np.clip(1.2 / (np.std(ln_x) + 1e-9), 0.5, 5.0))
    eta  = float(np.mean(data) / math.gamma(1 + 1/beta))
    return beta, max(eta, 1.0)



# MachineModel  —  Paper Section 3.2


class MachineModel:
    """
    Models one processing machine stage using a 3-state semi-Markov chain.

    States:
        2 = Good   (operating fine)
        1 = Degraded
        0 = Failed  (absorbing)

    Dwell time in each state ~ Weibull(beta, eta) fitted from AI4I TTF data.
    """
    STATES  = [0, 1, 2]
    LABELS  = ["Failed", "Degraded", "Good"]

    def __init__(self, stage, beta, eta):
        self.stage = stage
        self.beta  = beta
        self.eta   = eta

    def state_probs_over_time(self, t_max, dt=1.0):
        """
        Propagate state probability vector p(t) step by step.
        Returns (times, probs) where probs[s] is a list over time.

        Stay probability   Eq.(7):  p_xx  = (1-F(t+dt))/(1-F(t))
        Transit probability Eq.(8): p_x,x-1 = (F(t+dt)-F(t))/(1-F(t))
        """
        n_steps = int(t_max / dt) + 1
        times   = [i * dt for i in range(n_steps)]

        # Start in Good state
        p = [0.0, 0.0, 1.0]
        history = [[v] for v in p]

        for k in range(1, n_steps):
            t = k * dt
            Ft   = weibull_cdf(t,    self.beta, self.eta)
            Ftdt = weibull_cdf(t+dt, self.beta, self.eta)
            denom = 1.0 - Ft if (1.0 - Ft) > 1e-9 else 1e-9

            p_stay    = (1.0 - Ftdt) / denom
            p_transit = (Ftdt - Ft)  / denom
            p_stay    = max(0.0, min(1.0, p_stay))
            p_transit = max(0.0, min(1.0, p_transit))

            new_p = [0.0, 0.0, 0.0]
            new_p[0] = p[0]                       # Failed stays Failed
            new_p[2] = p[2] * p_stay              # Good stays Good
            new_p[1] = p[1] * p_stay + p[2] * p_transit   # Degraded
            new_p[0]+= p[1] * p_transit           # Degraded → Failed

            # Normalise for numerical stability
            s = sum(new_p)
            p = [v/s for v in new_p] if s > 0 else new_p

            for j in range(3):
                history[j].append(p[j])

        return times, history

    def survival(self, t):
        """Plain survival r(t)  →  Eq.(22)"""
        return weibull_survival(t, self.beta, self.eta)

    def survival_after_maintenance(self, tm, b):
        """
        Kijima II imperfect maintenance.  Eqs.(23)-(24).
        b = age regression factor: 0=perfect, 1=no effect.

        For beta<1 (infant-mortality), raw Kijima can reduce survival.
        We add a linear maintenance benefit so higher level always helps:
          r_final = r_base + (1-b) * (1 - r_base) * 0.5
        """
        r_base = self.survival(tm)
        # (1-b) is 0 when level=0, 1 when level=MC
        maintenance_gain = (1.0 - b) * (1.0 - r_base) * 0.5
        return min(1.0, r_base + maintenance_gain)



# QualityModel  —  Paper Section 3.3


class QualityModel:
    """
    WIP pass rate Q_i(t) for each stage.

    Fitted from AI4I data so that:
      Q(0) = 1.0   (perfect quality at start)
      Q(T) = 1 - failure_rate   (degrades over the task period)

    Simplified Eq.(17):  Q(t) = max(0,  1 - nu_rate * t^2)
    """

    def __init__(self, nu_rates):
        """nu_rates: dict {stage: float}"""
        self.nu_rates = nu_rates

    def Q(self, stage, t):
        """Pass rate at time t.  Eq.(17) simplified."""
        return max(0.0, 1.0 - self.nu_rates[stage] * t**2)

    def min_input(self, t):
        """
        Minimum input flow needed at each stage.  Eq.(10):
          f_i = TASK_DEMAND / product(Q_j for j >= i)
        """
        flows = {}
        for i, stage in enumerate(cfg.STAGES):
            downstream = 1.0
            for s in cfg.STAGES[i:]:
                downstream *= max(self.Q(s, t), 1e-9)
            flows[stage] = cfg.TASK_DEMAND / downstream
        return flows

    @classmethod
    def from_data(cls, df):
        """Fit nu_rate per stage from failure rates in the dataset."""
        nu_rates = {}
        for stage in cfg.STAGES:
            sub       = df[df["stage"] == stage]
            fail_rate = float(sub[cfg.FAILURE_COL].mean())
            # Q(T) = 1 - fail_rate  →  nu_rate = fail_rate / T^2
            nu_rates[stage] = fail_rate / max(cfg.TASK_HOURS**2, 1)
        return cls(nu_rates)



# MaintenanceModel  —  Paper Section 4.2


class MaintenanceModel:
    """
    Objective: Maximise R_sys = product of R_l(m) over all machines.  Eq.(27)
    Subject to:  total cost <= C0   Eq.(29)
                 total time <= T0   Eq.(30)

    Decision variable:  g[i] in {0, 1, 2, 3, 4, 5}  per stage
      0  = no maintenance
      5  = perfect maintenance
    """

    def __init__(self, machines):
        self.machines = machines   # list of MachineModel

    def cost(self, i, level):
        """C_{i,m}  Eq.(18)"""
        if level == 0:
            return 0.0
        return cfg.C_FIX[i] + (level / cfg.MC) * cfg.C_MAX[i]

    def time(self, i, level):
        """T_{i,m}  Eq.(20)"""
        if level == 0:
            return 0.0
        return cfg.T_FIX[i] + (level / cfg.MC) * cfg.T_MAX[i]

    def age_regression(self, level):
        """b = 1 - level/MC  (0 = perfect, 1 = nothing)  Eq.(23)"""
        return 1.0 - level / cfg.MC

    def R_machine(self, i, level):
        """Machine robustness after maintenance  Eqs.(23-25)"""
        b = self.age_regression(level)
        r = self.machines[i].survival_after_maintenance(cfg.NEXT_TASK_HOURS, b)
        return r   # working machine (s_break = 1)

    def R_sys(self, g):
        """System robustness  Eq.(27)"""
        result = 1.0
        for i, level in enumerate(g):
            result *= self.R_machine(i, level)
        return result

    def total_cost(self, g):
        return sum(self.cost(i, g[i]) for i in range(len(g)))

    def total_time(self, g):
        return sum(self.time(i, g[i]) for i in range(len(g)))

    def feasible(self, g):
        return self.total_cost(g) <= cfg.BUDGET_COST and \
               self.total_time(g) <= cfg.BUDGET_TIME

    def objective(self, g_continuous):
        """
        For the optimiser: MINIMISE negative robustness + constraint penalty.
        """
        g   = [min(cfg.MC, max(0, round(x))) for x in g_continuous]
        cv  = max(0.0, self.total_cost(g) - cfg.BUDGET_COST) / cfg.BUDGET_COST
        tv  = max(0.0, self.total_time(g) - cfg.BUDGET_TIME) / cfg.BUDGET_TIME
        pen = 2.0 * (cv + tv)
        return -(self.R_sys(g) - pen)

    def decode(self, g_continuous):
        return [min(cfg.MC, max(0, round(x))) for x in g_continuous]