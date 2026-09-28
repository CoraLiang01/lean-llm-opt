import gurobipy as gp
from gurobipy import GRB
r_Aalopuri = 20
I_Aalopuri = 10440
d_Aalopuri = 1483
m = gp.Model('Aalopuri_Revenue')
x = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x')
m.setObjective(r_Aalopuri * x, GRB.MAXIMIZE)
m.addConstr(x <= I_Aalopuri, name='inv')
m.addConstr(x <= d_Aalopuri, name='dem')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')