LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv",\n    "values": {\n      "Product Name": "Baby Food_255.28",\n      "Revenue": "255.28",\n      "Demand": "765850",\n      "Initial Inventory": "5627060"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
baby_products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in records:
    vals = rec['values']
    if 'Product Name' in vals and vals['Product Name'].startswith('Baby'):
        pname = vals['Product Name']
        baby_products.append(pname)
        try:
            revenue[pname] = float(vals['Revenue'])
            demand[pname] = int(vals['Demand'])
            initial_inventory[pname] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pname}: {e}')
for pname in baby_products:
    if pname not in revenue or pname not in demand or pname not in initial_inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Baby_Fulfillment')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(baby_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in baby_products)), GRB.MAXIMIZE)
for p in baby_products:
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x_vars[p] <= initial_inventory[p], name=f'inventory_{p}')
    m.addConstr(x_vars[p] >= 0, name=f'nonneg_{p}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')