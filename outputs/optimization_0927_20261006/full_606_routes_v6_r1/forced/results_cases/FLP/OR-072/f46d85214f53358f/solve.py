import gurobipy as gp
from gurobipy import GRB
T = 24
r_t = [20, 18, 15, 15, 20, 30, 60, 70, 50, 55, 65, 75, 80, 70, 60, 55, 60, 75, 85, 70, 50, 40, 35, 25]
periods = list(range(1, T + 1))
m = gp.Model('BusCrewScheduling')
x_vars = m.addVars(periods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in periods)), GRB.MINIMIZE)
for t in periods:
    idxs = [(t - k - 1) % T + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[j] for j in idxs)) >= r_t[t - 1], name=f'cov{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')