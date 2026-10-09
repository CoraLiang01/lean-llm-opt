import gurobipy as gp
from gurobipy import GRB
products = ['Organic Fruits', 'Organic Staples', 'Organic Vegetables']
revenue = {'Organic Fruits': 60.8, 'Organic Staples': 918.45, 'Organic Vegetables': 77.52}
demand = {'Organic Fruits': 678906, 'Organic Staples': 749927, 'Organic Vegetables': 699808}
initial_inventory = {'Organic Fruits': 5034020.0, 'Organic Staples': 5589290.0, 'Organic Vegetables': 5202710.0}
for p in products:
    if p not in revenue or p not in demand or p not in initial_inventory:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('Organ_Product_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= initial_inventory[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')