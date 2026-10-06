import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    periods = list(range(1, 25))
    r = {1: 20, 2: 18, 3: 15, 4: 15, 5: 20, 6: 30, 7: 60, 8: 70, 9: 50, 10: 55, 11: 65, 12: 75, 13: 80, 14: 70, 15: 60, 16: 55, 17: 60, 18: 75, 19: 85, 20: 70, 21: 50, 22: 40, 23: 35, 24: 25}
    if set(periods) != set(r.keys()):
        raise ValueError('Missing or extra period requirements in r.')
    m = gp.Model('BusCrewScheduling')
    x = m.addVars(periods, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[t] for t in periods)), GRB.MINIMIZE)
    for t in periods:
        idx = [(t - i - 1) % 24 + 1 for i in range(4)]
        m.addConstr(gp.quicksum((x[j] for j in idx)) >= r[t], name=f'c{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')