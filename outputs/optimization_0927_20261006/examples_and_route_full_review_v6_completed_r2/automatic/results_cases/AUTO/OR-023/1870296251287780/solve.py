LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10000463", "Revenue": "4.0", "Demand": "295", "Initial Inventory": "2000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10000487", "Revenue": "14.0", "Demand": "1002", "Initial Inventory": "7000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10003333", "Revenue": "14.0", "Demand": "958", "Initial Inventory": "7000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10009012", "Revenue": "4.0", "Demand": "777", "Initial Inventory": "6000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10009999", "Revenue": "4.0", "Demand": "271", "Initial Inventory": "2000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10011234", "Revenue": "4.0", "Demand": "244", "Initial Inventory": "2000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10027456", "Revenue": "14.0", "Demand": "990", "Initial Inventory": "7000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv", "values": {"Product_Reference": "ELE-SMA-10028567", "Revenue": "14.0", "Demand": "1000", "Initial Inventory": "7000.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10000463', 'Revenue': '4.0', 'Demand': '295', 'Initial Inventory': '2000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10000487', 'Revenue': '14.0', 'Demand': '1002', 'Initial Inventory': '7000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10003333', 'Revenue': '14.0', 'Demand': '958', 'Initial Inventory': '7000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10009012', 'Revenue': '4.0', 'Demand': '777', 'Initial Inventory': '6000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10009999', 'Revenue': '4.0', 'Demand': '271', 'Initial Inventory': '2000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10011234', 'Revenue': '4.0', 'Demand': '244', 'Initial Inventory': '2000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10027456', 'Revenue': '14.0', 'Demand': '990', 'Initial Inventory': '7000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv', 'values': {'Product_Reference': 'ELE-SMA-10028567', 'Revenue': '14.0', 'Demand': '1000', 'Initial Inventory': '7000.0'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    prod = vals['Product_Reference']
    products.append(prod)
    try:
        revenue[prod] = float(vals['Revenue'])
        demand[prod] = int(float(vals['Demand']))
        initial_inventory[prod] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for product {prod}: {e}')
upper_bounds = {prod: min(demand[prod], initial_inventory[prod]) for prod in products}
for prod in products:
    if prod not in revenue or prod not in demand or prod not in initial_inventory:
        raise ValueError(f'Missing data for product {prod}')
m = gp.Model('ELE_S_Fulfillment')
x_vars = m.addVars(products, lb=0, ub=[upper_bounds[prod] for prod in products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[prod] * x_vars[prod] for prod in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')