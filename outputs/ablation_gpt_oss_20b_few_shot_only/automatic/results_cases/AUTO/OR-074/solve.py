import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if 'Time Requirement' not in df.columns:
        raise ValueError("CSV does not contain 'Time Requirement' column")
    try:
        requirements = [int(v.strip()) for v in df['Time Requirement']]
    except Exception as e:
        raise ValueError(f'Error converting requirements to integers: {e}')
    if len(requirements) != 48:
        raise ValueError(f'Expected 48 periods, got {len(requirements)}')
    if any((r < 0 for r in requirements)):
        raise ValueError('Negative requirement found in data')
    T = range(1, 49)
    R = {s: requirements[s - 1] for s in T}
    cover = {}
    for s in T:
        cover[s] = [t for t in T if (s - t + 48) % 48 <= 15]
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    S = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((S[t] for t in T)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((S[t] for t in cover[s])) >= R[s] for s in T), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')