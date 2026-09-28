import gurobipy as gp
from gurobipy import GRB
products = ['ELE-SMA-10000463', 'ELE-SMA-10000487', 'ELE-SMA-10003333', 'ELE-SMA-10009012', 'ELE-SMA-10009999', 'ELE-SMA-10011234', 'ELE-SMA-10027456', 'ELE-SMA-10028567']
revenue = {'ELE-SMA-10000463': 4.0, 'ELE-SMA-10000487': 14.0, 'ELE-SMA-10003333': 14.0, 'ELE-SMA-10009012': 4.0, 'ELE-SMA-10009999': 4.0, 'ELE-SMA-10011234': 4.0, 'ELE-SMA-10027456': 14.0, 'ELE-SMA-10028567': 14.0}
demand = {'ELE-SMA-10000463': 295, 'ELE-SMA-10000487': 1002, 'ELE-SMA-10003333': 958, 'ELE-SMA-10009012': 777, 'ELE-SMA-10009999': 271, 'ELE-SMA-10011234': 244, 'ELE-SMA-10027456': 990, 'ELE-SMA-10028567': 1000}
initial_inventory = {'ELE-SMA-10000463': 2000.0, 'ELE-SMA-10000487': 7000.0, 'ELE-SMA-10003333': 7000.0, 'ELE-SMA-10009012': 6000.0, 'ELE-SMA-10009999': 2000.0, 'ELE-SMA-10011234': 2000.0, 'ELE-SMA-10027456': 7000.0, 'ELE-SMA-10028567': 7000.0}
for i in products:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for product {i}')
upper_bound = {i: min(demand[i], initial_inventory[i]) for i in products}
m = gp.Model('ELE_S_Fulfillment')
x = m.addVars(products, lb=0, ub=[upper_bound[i] for i in products], vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in products:
        print(f'x[{i}]: {x[i].X}')
else:
    print(f'Solver status: {m.Status}')