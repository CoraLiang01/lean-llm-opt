import gurobipy as gp
from gurobipy import GRB
revenue_per_unit = 20
demand_Aalopuri = 1483
initial_inventory_Aalopuri = 10440.0
m = gp.Model('Aalopuri_Revenue_Maximization')
x_vars = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x')
m.setObjective(revenue_per_unit * x_vars, GRB.MAXIMIZE)
m.addConstr(x_vars <= demand_Aalopuri, name='demand')
m.addConstr(x_vars <= initial_inventory_Aalopuri, name='inventory')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')