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
    product = vals['Product Name']
    if product.startswith('Baby'):
        baby_products.append(product)
        try:
            revenue[product] = float(vals['Revenue'])
            demand[product] = int(vals['Demand'])
            initial_inventory[product] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Missing or invalid data for product {product}: {e}')
for product in baby_products:
    if product not in revenue or product not in demand or product not in initial_inventory:
        raise ValueError(f'Missing data for product {product}')
m = gp.Model('Europe_Baby_Revenue')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(baby_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in baby_products)), GRB.MAXIMIZE)
for p in baby_products:
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x_vars[p] <= initial_inventory[p], name=f'inventory_{p}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')