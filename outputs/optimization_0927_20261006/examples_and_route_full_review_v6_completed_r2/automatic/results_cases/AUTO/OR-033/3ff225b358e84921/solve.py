LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv",\n    "values": {\n      "Product Name": "Baby Food_255.28",\n      "Revenue": "255.28",\n      "Demand": "765850",\n      "Initial Inventory": "5627060"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
product_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name', '').startswith('Baby')]
if not product_records:
    raise ValueError("No 'Baby' product records found in LEGACY_RECORDS.")
product_names = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in product_records:
    vals = rec['values']
    pname = vals['Product Name']
    product_names.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        initial_inventory[pname] = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pname}: {e}')
m = gp.Model('Baby_Product_Fulfillment')
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= demand[p] for p in product_names), name='')
m.addConstrs((x_vars[p] <= initial_inventory[p] for p in product_names), name='')
m.addConstrs((x_vars[p] >= 0 for p in product_names), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')