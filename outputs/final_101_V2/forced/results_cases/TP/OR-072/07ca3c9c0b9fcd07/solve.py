import gurobipy as gp
from gurobipy import GRB
periods = list(range(1, 25))
r = {1: 20, 2: 18, 3: 15, 4: 15, 5: 20, 6: 30, 7: 60, 8: 70, 9: 50, 10: 55, 11: 65, 12: 75, 13: 80, 14: 70, 15: 60, 16: 55, 17: 60, 18: 75, 19: 85, 20: 70, 21: 50, 22: 40, 23: 35, 24: 25}
if set(periods) != set(r.keys()):
    raise ValueError('Missing or extra r_s data for some periods.')
m = gp.Model('BusCrewScheduling')
x = m.addVars(periods, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), GRB.MINIMIZE)
for s in periods:
    covered = gp.quicksum((x[(s - k - 1) % 24 + 1] for k in range(4)))
    m.addConstr(covered >= r[s], name=f'cov{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')