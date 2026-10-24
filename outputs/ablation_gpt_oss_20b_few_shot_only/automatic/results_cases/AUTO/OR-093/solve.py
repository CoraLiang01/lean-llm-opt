import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    machines = df['Machine'].tolist()
    tasks = [c for c in df.columns if c != 'Machine']
    for t in tasks:
        df[t] = df[t].astype(float)
    cost = {}
    for (_, row) in df.iterrows():
        m = row['Machine']
        for t in tasks:
            cost[m, t] = row[t]
    for m in machines:
        for t in tasks:
            if (m, t) not in cost:
                raise ValueError(f'Missing cost for machine {m} task {t}')
    m = gurobipy.Model('Assignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m, t) for m in machines for t in tasks], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[m, t] * x[m, t] for m in machines for t in tasks)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[m, t] for t in tasks)) == 1 for m in machines), name='')
    m.addConstrs((gurobipy.quicksum((x[m, t] for m in machines)) == 1 for t in tasks), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.Status)