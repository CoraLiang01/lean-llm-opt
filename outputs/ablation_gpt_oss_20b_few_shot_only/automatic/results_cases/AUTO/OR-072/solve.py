import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['Shift'] = pd.to_numeric(df['Shift'], errors='raise')
    df['Number Required'] = pd.to_numeric(df['Number Required'], errors='raise')
    r_t = dict(zip(df['Shift'], df['Number Required']))
    required_t = set(range(1, 25))
    if not required_t.issubset(r_t.keys()):
        missing = required_t - r_t.keys()
        raise ValueError(f'Missing required data for shifts: {sorted(missing)}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    T = list(range(1, 25))
    x = m.addVars(T, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((x[t] for t in T)), GRB.MINIMIZE)
    for t in T:
        indices = [(t - k - 1) % 24 + 1 for k in range(4)]
        m.addConstr(gurobipy.quicksum((x[i] for i in indices)) >= r_t[t], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()