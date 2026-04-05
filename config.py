# config.py

# --- General System Parameters ---
NUM_MACHINES = 4
TASK_PERIOD = 40  # Days the system ran before needing maintenance [cite: 637]
NEXT_TASK_TIME = 50  # Duration of the upcoming mission [cite: 645]
DAILY_DEMAND = 200  # Units required per day [cite: 637]

# --- Maintenance Constraints ---
# These are the limits the optimizer must respect [cite: 645]
TOTAL_BUDGET = 6000  # Maximum money allowed for all repairs
TOTAL_TIME_LIMIT = 3.0  # Maximum hours allowed for all repairs

# --- Machine-Specific Data (Table 1 & 6 from Paper) ---
# Each dictionary represents a machine's name, job, and repair costs.
MACHINES_INFO = {
    "M1": {
        "process": "External Grinding",
        "fixed_cost": 160,
        "max_cost": 3500,
        "fixed_time": 0.16,
        "max_time": 1.86
    },
    "M2": {
        "process": "Face Grinding",
        "fixed_cost": 140,
        "max_cost": 3000,
        "fixed_time": 0.14,
        "max_time": 1.74
    },
    "M3": {
        "process": "Cementation",
        "fixed_cost": 110,
        "max_cost": 2500,
        "fixed_time": 0.11,
        "max_time": 1.60
    },
    "M4": {
        "process": "Wire Winding",
        "fixed_cost": 120,
        "max_cost": 2800,
        "fixed_time": 0.12,
        "max_time": 1.72
    }
}

# --- Weibull Degradation Parameters (Table 2 & Page 12) ---
# These 'math constants' define how quickly each machine wears out. [cite: 627, 630]
WEIBULL_PARAMS = {
    "M1": {"A": 0.3877, "Theta": 0.2998, "J": 2.2569},
    "M2": {"A": 0.4522, "Theta": 0.2271, "J": 2.1478},
    "M3": {"A": 0.3477, "Theta": 0.2428, "J": 1.9845},
    "M4": {"A": 0.3562, "Theta": 0.2831, "J": 2.1096}
}