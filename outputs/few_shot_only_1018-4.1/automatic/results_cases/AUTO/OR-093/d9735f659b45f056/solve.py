import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
machines = [row.strip() for row in df.iloc[:, 0]]
tasks = [col.strip() for col in df.columns[1:]]
cost = {}
for i, machine in enumerate(machines):
    cost[machine] = {}
    for j, task in enumerate(tasks):
        val = df.iloc[i, j + 1]
        try:
            cost[machine][task] = float(val)
        except Exception:
            raise ValueError(f"Missing or invalid cost for machine {machine}, task {task}: '{val}'")
if set(cost.keys()) != set(machines):
    raise ValueError('Mismatch in machine identifiers between CSV and cost dictionary.')
for machine in machines:
    if set(cost[machine].keys()) != set(tasks):
        raise ValueError(f'Mismatch in task identifiers for machine {machine}.')
m = gp.Model('AP1_12x12')
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