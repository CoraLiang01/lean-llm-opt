import gurobipy as gp
from gurobipy import GRB
revenue_per_unit = 20
demand = 1483
initial_inventory = 10440.0
m = gp.Model('Aalopuri_Revenue_Max')
x = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x')
m.setObjective(revenue_per_unit * x, GRB.MAXIMIZE)
m.addConstr(x <= demand, name='demand')
m.addConstr(x <= initial_inventory, name='inventory')
m.addConstr(x >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')