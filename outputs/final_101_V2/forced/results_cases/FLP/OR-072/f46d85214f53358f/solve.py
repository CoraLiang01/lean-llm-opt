import gurobipy as gp
from gurobipy import GRB
T = 24
r = {1: 20, 2: 18, 3: 15, 4: 15, 5: 20, 6: 30, 7: 60, 8: 70, 9: 50, 10: 55, 11: 65, 12: 75, 13: 80, 14: 70, 15: 60, 16: 55, 17: 60, 18: 75, 19: 85, 20: 70, 21: 50, 22: 40, 23: 35, 24: 25}
if set(r.keys()) != set(range(1, T + 1)):
    raise ValueError('r must have exactly one value for each t in 1..24')
m = gp.Model('BusCrewScheduling')
x = m.addVars(range(1, T + 1), lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((x[t] for t in range(1, T + 1))), GRB.MINIMIZE)
for t in range(1, T + 1):
    indices = [(t - i - 1) % T + 1 for i in range(4)]
    m.addConstr(gp.quicksum((x[j] for j in indices)) >= r[t], name=f'cover_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')