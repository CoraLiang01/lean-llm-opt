import pandas as pd
import gurobipy as gp
from gurobipy import GRB
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
machines = list(df.iloc[:, 0])
tasks = list(df.columns[1:])
cost = {}
for i, machine in enumerate(machines):
    cost[machine] = {}
    for j, task in enumerate(tasks):
        val = df.iloc[i, j + 1]
        if val == '':
            raise ValueError(f'Missing cost for machine {machine}, task {task}')
        try:
            cost[machine][task] = float(val)
        except Exception:
            raise ValueError(f'Non-numeric cost for machine {machine}, task {task}: {val}')
m = gp.Model('AP1_OriginalQuery')
x_vars = m.addVars(machines, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in tasks)) == 1 for i in machines), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in machines)) == 1 for j in tasks), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')