LEGACY_OBSERVATION = 'Sub Category,Revenue,Demand,Initial Inventory\nOrganic Fruits,60.8,678906,5034020.0\nOrganic Staples,918.45,749927,5589290.0\nOrganic Vegetables,77.52,699808,5202710.0'
LEGACY_RECORDS = [{'source': '', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    prod = vals['Sub Category']
    products.append(prod)
    try:
        revenue[prod] = float(vals['Revenue'])
        demand[prod] = int(float(vals['Demand']))
        inventory[prod] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for product {prod}: {e}')
for prod in products:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f'Missing data for product {prod}')
upper_bound = {prod: min(demand[prod], inventory[prod]) for prod in products}
m = gp.Model('Organ_Fulfillment')
x = m.addVars(products, lb=0, ub=upper_bound, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[prod] * x[prod] for prod in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')