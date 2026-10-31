# model9.py
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os

# -------- load parameters from CSV --------
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
param_path = os.path.join(script_dir, "parameters.csv")  
df = pd.read_csv(param_path)

# sets
trucks = df["truck_id"].astype(int).tolist()
periods = [1, 2, 3, 4]

# parameters by dict
Q = dict(zip(df["truck_id"], df["Q"]))
S = dict(zip(df["truck_id"], df["S"]))
C = dict(zip(df["truck_id"], df["C"]))

# Read demand (all rows identical; take the first)
d_row = df.iloc[0][["d1","d2","d3","d4"]].to_list()
d = {t: d_row[t-1] for t in periods}

# -------- model --------
m = gp.Model("Truck_Transport_Optimization")

# Decision variables
x = m.addVars(trucks, periods, vtype=GRB.CONTINUOUS, lb=0.0, name="x")     # 10*4 = 40
y = m.addVars(trucks, periods, vtype=GRB.BINARY, name="y")                 # 10*4 = 40
u = m.addVars(trucks, [2,3,4], vtype=GRB.BINARY, name="u")                 # 10*3 = 30  --> total = 110

# objective: startup-in-period1 via S_i * y[i,1], startup t>=2 via S_i * u[i,t]
m.setObjective(
    gp.quicksum(S[i] * y[i, 1] for i in trucks) +
    gp.quicksum(S[i] * u[i, t] for i in trucks for t in [2,3,4]) +
    gp.quicksum(C[i] * x[i, t] for i in trucks for t in periods),
    GRB.MINIMIZE
)

# 1) Demand constraints
for t in periods:
    m.addConstr(gp.quicksum(x[i, t] for i in trucks) >= d[t], name=f"demand_{t}")

# 2) Capacity linkage
for i in trucks:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f"cap_{i}_{t}")

# 3) 10% reserve constraints
for t in periods:
    m.addConstr(
        gp.quicksum(x[i, t] for i in trucks) <= 0.9 * gp.quicksum(Q[i] * y[i, t] for i in trucks),
        name=f"reserve_{t}"
    )

# 4) Startup logic (t >= 2) + tightening constraints
for i in trucks:
    for t in [2,3,4]:
        m.addConstr(u[i, t] >= y[i, t] - y[i, t-1], name=f"start_lb_{i}_{t}")
        m.addConstr(u[i, t] <= y[i, t],             name=f"start_ub1_{i}_{t}")
        m.addConstr(u[i, t] <= 1 - y[i, t-1],       name=f"start_ub2_{i}_{t}")

# 5) Minimum up-time constraints
for i in trucks:
    # If started at t=1 (from off), must stay on at t=2
    m.addConstr(y[i, 1] + y[i, 2] >= 2 * y[i, 1], name=f"minup_t1_{i}")
    # For t=2,3: if u[i,t]=1, must be on in t and t+1
    for t in [2,3]:
        m.addConstr(y[i, t] + y[i, t+1] >= 2 * u[i, t], name=f"minup_{i}_{t}")
    # Forbid starting at the last period (no room to satisfy min-up)
    m.addConstr(u[i, 4] == 0, name=f"no_start_last_{i}")

# 6) Per-truck ramp constraints (load fluctuation)
for i in trucks:
    for t in [2,3,4]:
        m.addConstr(x[i, t] - x[i, t-1] <= 300, name=f"ramp_up_{i}_{t}")
        m.addConstr(x[i, t-1] - x[i, t] <= 300, name=f"ramp_dn_{i}_{t}")

# Optimize
m.optimize()

# Export model
m.write(os.path.join(script_dir, "model.lp"))

# (Optional) Validate variable count = 110 after model is built
print(f"Total variables: {m.NumVars}")
print(f"Total constraints: {m.NumConstrs}")



