import gurobipy as gp
from gurobipy import GRB
products = ['Organic Fruits', 'Organic Staples', 'Organic Vegetables']
revenue = {'Organic Fruits': 60.8, 'Organic Staples': 918.45, 'Organic Vegetables': 77.52}
inventory = {'Organic Fruits': 5034020.0, 'Organic Staples': 5589290.0, 'Organic Vegetables': 5202710.0}
demand = {'Organic Fruits': 678906, 'Organic Staples': 749927, 'Organic Vegetables': 699808}
for p in products:
    if p not in revenue or p not in inventory or p not in demand:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('Organ_Revenue_Max')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] <= inventory[p], name=f'inv_{p}')
    m.addConstr(x_vars[p] <= demand[p], name=f'dem_{p}')
    m.addConstr(x_vars[p] >= 0, name=f'nonneg_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')