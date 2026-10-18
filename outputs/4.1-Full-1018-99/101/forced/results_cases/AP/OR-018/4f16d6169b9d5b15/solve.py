LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv",\n    "values": {\n      "Product Name": "Baby Food_255.28",\n      "Revenue": "255.28",\n      "Demand": "3066513",\n      "Initial Inventory": "22749210"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '3066513', 'Initial Inventory': '22749210'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
baby_products = []
revenues = {}
demands = {}
inventories = {}
for rec in records:
    vals = rec['values']
    if 'Product Name' in vals and vals['Product Name'].startswith('Baby'):
        prod = vals['Product Name']
        baby_products.append(prod)
        revenues[prod] = float(vals['Revenue'])
        demands[prod] = int(vals['Demand'])
        inventories[prod] = int(vals['Initial Inventory'])
for prod in baby_products:
    if prod not in revenues or prod not in demands or prod not in inventories:
        raise ValueError(f'Missing data for product {prod}')
m = gp.Model('Baby_Revenue_Max')
x_vars = m.addVars(baby_products, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenues[prod] * x_vars[prod] for prod in baby_products)), GRB.MAXIMIZE)
for prod in baby_products:
    upper = min(inventories[prod], demands[prod])
    m.addConstr(x_vars[prod] >= 0, name=f'lb_{prod}')
    m.addConstr(x_vars[prod] <= upper, name=f'ub_{prod}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')