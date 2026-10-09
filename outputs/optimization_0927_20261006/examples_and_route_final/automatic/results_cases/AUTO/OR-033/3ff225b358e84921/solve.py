LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv", "values": {"Product Name": "Baby Food_255.28", "Revenue": "255.28", "Demand": "765850", "Initial Inventory": "5627060"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
baby_products = []
revenue = {}
initial_inventory = {}
demand = {}
for rec in records:
    vals = rec['values']
    if 'Baby' in vals.get('Product Name', ''):
        name = vals['Product Name']
        baby_products.append(name)
        try:
            revenue[name] = float(vals['Revenue'])
            initial_inventory[name] = int(vals['Initial Inventory'])
            demand[name] = int(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {name}: {e}')
for name in baby_products:
    if name not in revenue or name not in initial_inventory or name not in demand:
        raise ValueError(f'Missing data for product {name}')
m = gp.Model('Baby_Product_Fulfillment')
x_vars = m.addVars(baby_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[name] * x_vars[name] for name in baby_products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[name] <= initial_inventory[name] for name in baby_products), name='')
m.addConstrs((x_vars[name] <= demand[name] for name in baby_products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')