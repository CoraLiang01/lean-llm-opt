import gurobipy as gp
from gurobipy import GRB
Organ = ['Organic Fruits', 'Organic Staples', 'Organic Vegetables']
revenue = {'Organic Fruits': 60.8, 'Organic Staples': 918.45, 'Organic Vegetables': 77.52}
demand = {'Organic Fruits': 678906, 'Organic Staples': 749927, 'Organic Vegetables': 699808}
inventory = {'Organic Fruits': 5034020.0, 'Organic Staples': 5589290.0, 'Organic Vegetables': 5202710.0}
for i in Organ:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product: {i}')
m = gp.Model('Organ_Revenue_Max')
x = m.addVars(Organ, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in Organ)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in Organ), name='')
m.addConstrs((x[i] <= demand[i] for i in Organ), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')