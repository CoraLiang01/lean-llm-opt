import gurobipy as gp
from gurobipy import GRB
products = ['ELE-SMA-10000463', 'ELE-SMA-10000487', 'ELE-SMA-10003333', 'ELE-SMA-10009012', 'ELE-SMA-10009999', 'ELE-SMA-10011234', 'ELE-SMA-10027456', 'ELE-SMA-10028567']
revenue = {'ELE-SMA-10000463': 4.0, 'ELE-SMA-10000487': 14.0, 'ELE-SMA-10003333': 14.0, 'ELE-SMA-10009012': 4.0, 'ELE-SMA-10009999': 4.0, 'ELE-SMA-10011234': 4.0, 'ELE-SMA-10027456': 14.0, 'ELE-SMA-10028567': 14.0}
demand = {'ELE-SMA-10000463': 295, 'ELE-SMA-10000487': 1002, 'ELE-SMA-10003333': 958, 'ELE-SMA-10009012': 777, 'ELE-SMA-10009999': 271, 'ELE-SMA-10011234': 244, 'ELE-SMA-10027456': 990, 'ELE-SMA-10028567': 1000}
inventory = {'ELE-SMA-10000463': 2000.0, 'ELE-SMA-10000487': 7000.0, 'ELE-SMA-10003333': 7000.0, 'ELE-SMA-10009012': 6000.0, 'ELE-SMA-10009999': 2000.0, 'ELE-SMA-10011234': 2000.0, 'ELE-SMA-10027456': 7000.0, 'ELE-SMA-10028567': 7000.0}
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('ELE_S_Revenue_Max')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= inventory[p] for p in products), name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')