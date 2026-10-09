import gurobipy as gp
from gurobipy import GRB
products = ['Baby Food_255.28']
revenue = {'Baby Food_255.28': 255.28}
initial_inventory = {'Baby Food_255.28': 5627060}
demand = {'Baby Food_255.28': 765850}
for p in products:
    if p not in revenue or p not in initial_inventory or p not in demand:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Baby_Product_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= initial_inventory[p] for p in products), name='')
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')