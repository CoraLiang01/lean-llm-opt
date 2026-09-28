LEGACY_OBSERVATION = '[\n  {\n    "Product_Reference": "ELE-SMA-10000463",\n    "Revenue": "4.0",\n    "Demand": "295",\n    "Initial Inventory": "2000.0"\n  },\n  {\n    "Product_Reference": "ELE-SMA-10000487",\n    "Revenue": "14.0",\n    "Demand": "1002",\n    "Initial Inventory": "7000.0"\n  },\n  {\n    "Product_Reference": "ELE-SMA-10003333",\n    "Revenue": "14.0",\n    "Demand": "958",\n    "Initial Inventory": "7000.0"\n  },\n  {\n    "Product_Reference": "ELE-SMA-10009012",\n    "Revenue": "4.0",\n    "Demand": "777",\n    "Initial Inventory": "6000.0"\n  },\n  {\n    "Product_Reference": "ELE-SMA-10009999",\n    "Revenue": "4.0",\n    "Demand": "271",\n    "Initial Inventory": "2000.0"\n  },\n  {\n    "Product_Reference": "ELE-SMA-10011234",\n    "Revenue": "4.0",\n    "Demand": "244",\n    "Initial Inventory": "2000.0"\n  },\n  {\n    "Product_Reference": "ELE-SMA-10027456",\n    "Revenue": "14.0",\n    "Demand": "990",\n    "Initial Inventory": "7000.0"\n  },\n  {\n    "Product_Reference": "ELE-SMA-10028567",\n    "Revenue": "14.0",\n    "Demand": "1000",\n    "Initial Inventory": "7000.0"\n  }\n]'
LEGACY_RECORDS = [{'source': '', 'values': {'Product_Reference': 'ELE-SMA-10000463', 'Revenue': '4.0', 'Demand': '295', 'Initial Inventory': '2000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10000487', 'Revenue': '14.0', 'Demand': '1002', 'Initial Inventory': '7000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10003333', 'Revenue': '14.0', 'Demand': '958', 'Initial Inventory': '7000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10009012', 'Revenue': '4.0', 'Demand': '777', 'Initial Inventory': '6000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10009999', 'Revenue': '4.0', 'Demand': '271', 'Initial Inventory': '2000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10011234', 'Revenue': '4.0', 'Demand': '244', 'Initial Inventory': '2000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10027456', 'Revenue': '14.0', 'Demand': '990', 'Initial Inventory': '7000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10028567', 'Revenue': '14.0', 'Demand': '1000', 'Initial Inventory': '7000.0'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pid = vals['Product_Reference']
    products.append(pid)
    try:
        revenue[pid] = float(vals['Revenue'])
        demand[pid] = int(float(vals['Demand']))
        inventory[pid] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pid}: {e}')
for pid in products:
    if pid not in revenue or pid not in demand or pid not in inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('ELE_S_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
for i in products:
    m.addConstr(x[i] <= inventory[i], name=f'inv_{i}')
    m.addConstr(x[i] <= demand[i], name=f'dem_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')