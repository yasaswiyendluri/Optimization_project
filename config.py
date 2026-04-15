#  Dataset 
DATA_PATH = "data/ai4i2020.csv"

# AI4I column names
TYPE_COL    = "Type"               # L, M, H
WEAR_COL    = "Tool wear [min]"    # our KQC (quality characteristic)
TORQUE_COL  = "Torque [Nm]"       # noise factor
FAILURE_COL = "Machine failure"   # 0 or 1

# 3 machine types = 3 stages in series
STAGES = ["L", "M", "H"]

#  Task parameters  
TASK_HOURS      = 24    # one shift = 24 hours (each row = 1 hour)
NEXT_TASK_HOURS = 48    # next task duration tm
TASK_DEMAND     = 200   # units required per task

#  Maintenance budget 
BUDGET_COST = 8000   # C0
BUDGET_TIME = 4.0    # T0 (hours)

# Fixed + max variable maintenance cost/time per stage [L, M, H]
C_FIX = [120,  140,  160 ]
C_MAX = [2500, 3000, 3500]
T_FIX = [0.20, 0.30, 0.40]
T_MAX = [0.80, 1.00, 1.20]

# Maintenance levels: 0 = do nothing, MC = perfect
MC = 5

# PSO / ASA-PSO 
N_PARTICLES = 30
MAX_ITER    = 200
OMEGA       = 0.7    # inertia weight
C1          = 1.5    # cognitive
C2          = 1.5    # social
SA_MU       = 0.95   # SA cooling rate
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]