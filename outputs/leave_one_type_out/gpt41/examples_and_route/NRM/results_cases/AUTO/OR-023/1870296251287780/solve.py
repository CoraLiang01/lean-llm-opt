LEGACY_OBSERVATION = 'Product_Reference,Revenue,Demand,Initial Inventory\nELE-SMA-10000463,4.0,295,2000.0\nELE-SMA-10000487,14.0,1002,7000.0\nELE-SMA-10003333,14.0,958,7000.0\nELE-SMA-10009012,4.0,777,6000.0\nELE-SMA-10009999,4.0,271,2000.0\nELE-SMA-10011234,4.0,244,2000.0\nELE-SMA-10027456,14.0,990,7000.0\nELE-SMA-10028567,14.0,1000,7000.0'
LEGACY_RECORDS = [{'source': '', 'values': {'Product_Reference': 'ELE-SMA-10000463', 'Revenue': '4.0', 'Demand': '295', 'Initial Inventory': '2000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10000487', 'Revenue': '14.0', 'Demand': '1002', 'Initial Inventory': '7000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10003333', 'Revenue': '14.0', 'Demand': '958', 'Initial Inventory': '7000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10009012', 'Revenue': '4.0', 'Demand': '777', 'Initial Inventory': '6000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10009999', 'Revenue': '4.0', 'Demand': '271', 'Initial Inventory': '2000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10011234', 'Revenue': '4.0', 'Demand': '244', 'Initial Inventory': '2000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10027456', 'Revenue': '14.0', 'Demand': '990', 'Initial Inventory': '7000.0'}}, {'source': '', 'values': {'Product_Reference': 'ELE-SMA-10028567', 'Revenue': '14.0', 'Demand': '1000', 'Initial Inventory': '7000.0'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    prod = vals['Product_Reference']
    products.append(prod)
    try:
        revenue[prod] = float(vals['Revenue'])
        demand[prod] = int(float(vals['Demand']))
        inventory[prod] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {prod}: {e}')
for prod in products:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f'Missing data for product {prod}')
m = gp.Model('Supermarket_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
for i in products:
    m.addConstr(x[i] <= inventory[i], name=f'cap_{i}')
    m.addConstr(x[i] <= demand[i], name=f'dem_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')