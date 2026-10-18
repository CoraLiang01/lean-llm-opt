LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv",\n    "values": {\n      "Product Name": "Baby Food_255.28",\n      "Revenue": "255.28",\n      "Demand": "765850",\n      "Initial Inventory": "5627060"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
for rec in records:
    vals = rec['values']
    if vals.get('Product Name', '').startswith('Baby'):
        products.append({'Product Name': vals['Product Name'], 'Revenue': float(vals['Revenue']), 'Demand': int(vals['Demand']), 'Initial Inventory': int(vals['Initial Inventory'])})
if len(products) == 0:
    raise ValueError("No 'Baby' products found in LEGACY_RECORDS.")
product_names = [p['Product Name'] for p in products]
revenue = {p['Product Name']: p['Revenue'] for p in products}
demand = {p['Product Name']: p['Demand'] for p in products}
inventory = {p['Product Name']: p['Initial Inventory'] for p in products}
m = gp.Model('Europe_Baby_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
for p in product_names:
    m.addConstr(x_vars[p] <= inventory[p], name=f'inventory_{p}')
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')