import gurobipy as gp
from gurobipy import GRB
models = [f'HiFi-{i}' for i in range(1, 102)]
workstations = [1, 2, 3]
t = {1: [6, 4, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 10], 2: [5, 5, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 3], 3: [4, 6, 5, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6]}
for w in workstations:
    if len(t[w]) != 101:
        raise ValueError(f'Processing time data for workstation {w} does not have 101 entries.')
C = {1: 1440, 2: 1440, 3: 1440}
p = {1: 0.1, 2: 0.14, 3: 0.12}
E = {w: C[w] * (1 - p[w]) for w in workstations}
t_wm = {}
for w in workstations:
    t_wm[w] = {}
    for idx, m in enumerate(models):
        t_wm[w][m] = t[w][idx]
m = gp.Model('HiFi_IdleTime_Min')
x = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
idle_expr = gp.LinExpr()
for w in workstations:
    idle_expr += E[w]
    idle_expr -= gp.quicksum((t_wm[w][m] * x[m] for m in models))
m.setObjective(idle_expr, GRB.MINIMIZE)
for w in workstations:
    m.addConstr(gp.quicksum((t_wm[w][m] * x[m] for m in models)) <= E[w], name=f'cap{w}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')