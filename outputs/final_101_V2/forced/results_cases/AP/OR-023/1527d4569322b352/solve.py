import gurobipy as gp
from gurobipy import GRB
products = ['ELE-SMA-10000463', 'ELE-SMA-10000487', 'ELE-SMA-10003333', 'ELE-SMA-10009012', 'ELE-SMA-10009999', 'ELE-SMA-10011234', 'ELE-SMA-10027456', 'ELE-SMA-10028567']
revenue = {'ELE-SMA-10000463': 4.0, 'ELE-SMA-10000487': 14.0, 'ELE-SMA-10003333': 14.0, 'ELE-SMA-10009012': 4.0, 'ELE-SMA-10009999': 4.0, 'ELE-SMA-10011234': 4.0, 'ELE-SMA-10027456': 14.0, 'ELE-SMA-10028567': 14.0}
initial_inventory = {'ELE-SMA-10000463': 2000, 'ELE-SMA-10000487': 7000, 'ELE-SMA-10003333': 7000, 'ELE-SMA-10009012': 6000, 'ELE-SMA-10009999': 2000, 'ELE-SMA-10011234': 2000, 'ELE-SMA-10027456': 7000, 'ELE-SMA-10028567': 7000}
demand = {'ELE-SMA-10000463': 295, 'ELE-SMA-10000487': 1002, 'ELE-SMA-10003333': 958, 'ELE-SMA-10009012': 777, 'ELE-SMA-10009999': 271, 'ELE-SMA-10011234': 244, 'ELE-SMA-10027456': 990, 'ELE-SMA-10028567': 1000}
for p in products:
    if p not in revenue or p not in initial_inventory or p not in demand:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('ELE_S_Fulfillment')
x = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
for p in products:
    m.addConstr(x[p] <= initial_inventory[p], name=f'ub_inv_{p}')
    m.addConstr(x[p] <= demand[p], name=f'ub_dem_{p}')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')