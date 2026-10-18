LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv", "values": {"Product Name": "Baby Food_255.28", "Revenue": "255.28", "Demand": "3066513", "Initial Inventory": "22749210"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '3066513', 'Initial Inventory': '22749210'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
baby_products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in records:
    vals = rec['values']
    product = vals['Product Name']
    baby_products.append(product)
    try:
        revenue[product] = float(vals['Revenue'])
        demand[product] = int(vals['Demand'])
        initial_inventory[product] = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} for product {product}')
for p in baby_products:
    if p not in revenue or p not in demand or p not in initial_inventory:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Baby_Product_Fulfillment')
x_vars = m.addVars(baby_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in baby_products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= demand[p] for p in baby_products), name='')
m.addConstrs((x_vars[p] <= initial_inventory[p] for p in baby_products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')