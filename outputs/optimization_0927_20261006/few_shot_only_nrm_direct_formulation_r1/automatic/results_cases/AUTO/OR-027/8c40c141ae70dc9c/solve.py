import gurobipy as gp
from gurobipy import GRB
items = ['Organic Fruits', 'Organic Staples', 'Organic Vegetables']
revenue = {'Organic Fruits': 60.8, 'Organic Staples': 918.45, 'Organic Vegetables': 77.52}
demand = {'Organic Fruits': 678906, 'Organic Staples': 749927, 'Organic Vegetables': 699808}
inventory = {'Organic Fruits': 5034020.0, 'Organic Staples': 5589290.0, 'Organic Vegetables': 5202710.0}
for i in items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for item: {i}')
m = gp.Model('Supermarket_Organ_MaxRevenue')
x_vars = m.addVars(items, lb=0, ub={i: min(demand[i], inventory[i]) for i in items}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')