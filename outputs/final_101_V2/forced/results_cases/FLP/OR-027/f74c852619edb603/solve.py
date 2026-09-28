import gurobipy as gp
from gurobipy import GRB
products = ['Organic Fruits', 'Organic Staples', 'Organic Vegetables']
revenue = {'Organic Fruits': 60.8, 'Organic Staples': 918.45, 'Organic Vegetables': 77.52}
initial_inventory = {'Organic Fruits': 5034020, 'Organic Staples': 5589290, 'Organic Vegetables': 5202710}
demand = {'Organic Fruits': 678906, 'Organic Staples': 749927, 'Organic Vegetables': 699808}
for p in products:
    if p not in revenue or p not in initial_inventory or p not in demand:
        raise ValueError(f'Missing data for product: {p}')
upper_bound = {p: min(initial_inventory[p], demand[p]) for p in products}
m = gp.Model('Supermarket_Organ_Revenue')
x = m.addVars(products, lb=0, ub=upper_bound, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')