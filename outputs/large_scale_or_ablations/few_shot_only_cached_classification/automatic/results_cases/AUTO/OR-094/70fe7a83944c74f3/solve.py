import gurobipy as gp
from gurobipy import GRB
models = [f'HiFi-{k}' for k in range(1, 102)]
workstations = ['1', '2', '3']
t = {'1': {'HiFi-1': 6, 'HiFi-2': 4, 'HiFi-101': 9}, '2': {'HiFi-1': 5, 'HiFi-2': 5, 'HiFi-101': 3}, '3': {'HiFi-1': 4, 'HiFi-2': 6, 'HiFi-101': 6}}
for i in workstations:
    for k in models:
        if k not in t[i]:
            raise ValueError(f'Missing processing time for workstation {i}, model {k}')
C = {'1': 1296, '2': 1238.4, '3': 1267.2}
m = gp.Model('Radio_IdleTime_Min')
x = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
idle_expr = gp.LinExpr()
for i in workstations:
    idle_expr += C[i] - gp.quicksum((t[i][k] * x[k] for k in models))
m.setObjective(idle_expr, GRB.MINIMIZE)
for i in workstations:
    m.addConstr(gp.quicksum((t[i][k] * x[k] for k in models)) <= C[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')