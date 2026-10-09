import gurobipy as gp
from gurobipy import GRB
r = [20, 18, 15, 15, 20, 30, 60, 70, 50, 55, 65, 75, 80, 70, 60, 55, 60, 75, 85, 70, 50, 40, 35, 25]
n = 24
periods = list(range(1, n + 1))
m = gp.Model('BusCrewScheduling')
x_vars = m.addVars(periods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in periods)), GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[(t - k - 1) % n + 1] for k in range(4))) >= r[t - 1], name=f'cover_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')