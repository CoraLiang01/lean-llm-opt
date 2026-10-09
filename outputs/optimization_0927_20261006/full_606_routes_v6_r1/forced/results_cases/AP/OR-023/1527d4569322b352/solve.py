import gurobipy as gp
from gurobipy import GRB
products = [{'Product_Reference': 'ELE-SMA-10000463', 'Revenue': 4.0, 'Demand': 295, 'Initial Inventory': 2000.0}, {'Product_Reference': 'ELE-SMA-10000487', 'Revenue': 14.0, 'Demand': 1002, 'Initial Inventory': 7000.0}, {'Product_Reference': 'ELE-SMA-10003333', 'Revenue': 14.0, 'Demand': 958, 'Initial Inventory': 7000.0}, {'Product_Reference': 'ELE-SMA-10009012', 'Revenue': 4.0, 'Demand': 777, 'Initial Inventory': 6000.0}, {'Product_Reference': 'ELE-SMA-10009999', 'Revenue': 4.0, 'Demand': 271, 'Initial Inventory': 2000.0}, {'Product_Reference': 'ELE-SMA-10011234', 'Revenue': 4.0, 'Demand': 244, 'Initial Inventory': 2000.0}, {'Product_Reference': 'ELE-SMA-10027456', 'Revenue': 14.0, 'Demand': 990, 'Initial Inventory': 7000.0}, {'Product_Reference': 'ELE-SMA-10028567', 'Revenue': 14.0, 'Demand': 1000, 'Initial Inventory': 7000.0}]
product_ids = ['ELE-SMA-10000463', 'ELE-SMA-10000487', 'ELE-SMA-10003333', 'ELE-SMA-10009012', 'ELE-SMA-10009999', 'ELE-SMA-10011234', 'ELE-SMA-10027456', 'ELE-SMA-10028567']
revenue = {}
demand = {}
initial_inventory = {}
upper_bound = {}
for p in products:
    pid = p['Product_Reference']
    revenue[pid] = float(p['Revenue'])
    demand[pid] = float(p['Demand'])
    initial_inventory[pid] = float(p['Initial Inventory'])
    upper_bound[pid] = min(initial_inventory[pid], demand[pid])
for pid in product_ids:
    if pid not in revenue or pid not in demand or pid not in initial_inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('ELE_S_Fulfillment')
x_vars = m.addVars(product_ids, lb=0, ub=[upper_bound[pid] for pid in product_ids], vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), GRB.MAXIMIZE)
for pid in product_ids:
    m.addConstr(x_vars[pid] <= upper_bound[pid], name=f'ub_{pid}')
    m.addConstr(x_vars[pid] >= 0, name=f'lb_{pid}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')