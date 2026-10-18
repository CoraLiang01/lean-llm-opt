LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv",\n    "values": {\n      "Product Name": "Baby Food_255.28",\n      "Revenue": "255.28",\n      "Demand": "765850",\n      "Initial Inventory": "5627060"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
baby_products = []
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'Product Name' in vals and 'Baby' in vals['Product Name']:
        baby_products.append(vals)
product_ids = []
revenue = {}
demand = {}
initial_inventory = {}
for prod in baby_products:
    pid = prod['Product Name']
    try:
        revenue[pid] = float(prod['Revenue'])
        demand[pid] = int(prod['Demand'])
        initial_inventory[pid] = int(prod['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Missing or invalid data for product {pid}: {e}')
    product_ids.append(pid)
if not set(product_ids) == set(revenue.keys()) == set(demand.keys()) == set(initial_inventory.keys()):
    raise ValueError('Mismatch in product identifiers or missing data.')
m = gp.Model('Baby_Product_Revenue_Maximization')
x_vars = m.addVars(product_ids, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), GRB.MAXIMIZE)
for pid in product_ids:
    m.addConstr(x_vars[pid] <= demand[pid], name=f'demand_{pid}')
    m.addConstr(x_vars[pid] <= initial_inventory[pid], name=f'inventory_{pid}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')