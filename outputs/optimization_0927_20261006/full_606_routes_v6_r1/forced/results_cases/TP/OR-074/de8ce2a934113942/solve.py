import gurobipy as gp
from gurobipy import GRB
periods = list(range(1, 49))
requirements = {1: 2, 2: 3, 3: 4, 4: 6, 5: 5, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 9, 13: 8, 14: 8, 15: 9, 16: 9, 17: 10, 18: 12, 19: 11, 20: 11, 21: 12, 22: 11, 23: 10, 24: 9, 25: 8, 26: 7, 27: 6, 28: 5, 29: 5, 30: 6, 31: 7, 32: 8, 33: 9, 34: 10, 35: 9, 36: 8, 37: 7, 38: 6, 39: 5, 40: 4, 41: 4, 42: 3, 43: 3, 44: 3, 45: 3, 46: 4, 47: 4, 48: 4}
if set(requirements.keys()) != set(periods):
    raise ValueError('Requirements data does not cover all periods 1..48.')
m = gp.Model('Waitstaff_Scheduling')
x_vars = m.addVars(periods, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in periods)), GRB.MINIMIZE)
for k in periods:
    covered_starts = [(k - i - 1) % 48 + 1 for i in range(16)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covered_starts)) >= requirements[k], name=f'cover_{k}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')