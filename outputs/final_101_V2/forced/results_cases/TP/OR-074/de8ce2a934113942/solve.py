import gurobipy as gp
from gurobipy import GRB
N = 48
periods = list(range(1, N + 1))
r = {1: 2, 2: 3, 3: 4, 4: 6, 5: 5, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 9, 13: 8, 14: 8, 15: 9, 16: 9, 17: 10, 18: 12, 19: 11, 20: 11, 21: 12, 22: 11, 23: 10, 24: 9, 25: 8, 26: 7, 27: 6, 28: 5, 29: 5, 30: 6, 31: 7, 32: 8, 33: 9, 34: 10, 35: 9, 36: 8, 37: 7, 38: 6, 39: 5, 40: 4, 41: 4, 42: 3, 43: 3, 44: 3, 45: 3, 46: 4, 47: 4, 48: 4}
if set(periods) != set(r.keys()):
    raise ValueError('r data missing or extra periods.')
m = gp.Model('Waitstaff_Shift_Scheduling')
x = m.addVars(periods, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), GRB.MINIMIZE)
for s in periods:
    expr = gp.quicksum((x[(s - k - 1) % N + 1] for k in range(16)))
    m.addConstr(expr >= r[s], name=f'cover_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')