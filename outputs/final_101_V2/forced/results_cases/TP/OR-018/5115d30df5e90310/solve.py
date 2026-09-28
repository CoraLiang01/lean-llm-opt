import gurobipy as gp
from gurobipy import GRB
products = ['Baby Food_255.28']
revenue = {'Baby Food_255.28': 255.28}
demand = {'Baby Food_255.28': 3066513}
initial_inventory = {'Baby Food_255.28': 22749210}
for p in products:
    if p not in revenue or p not in demand or p not in initial_inventory:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Baby_Product_Revenue')
x = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x')
m.setObjective(revenue['Baby Food_255.28'] * x, GRB.MAXIMIZE)
m.addConstr(x <= demand['Baby Food_255.28'], name='demand')
m.addConstr(x <= initial_inventory['Baby Food_255.28'], name='inventory')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')