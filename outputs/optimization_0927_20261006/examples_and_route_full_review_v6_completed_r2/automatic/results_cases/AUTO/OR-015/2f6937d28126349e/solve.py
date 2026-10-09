LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv",\n    "values": {\n      "Product Name": "Aalopuri",\n      "Revenue": "20",\n      "Demand": "1483",\n      "Initial Inventory": "10440.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
aalop_products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in records:
    vals = rec['values']
    if 'Product Name' in vals and vals['Product Name'].startswith('Aalop'):
        name = vals['Product Name']
        aalop_products.append(name)
        try:
            revenue[name] = float(vals['Revenue'])
            demand[name] = int(float(vals['Demand']))
            initial_inventory[name] = float(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for {name}: {e}')
for name in aalop_products:
    if name not in revenue or name not in demand or name not in initial_inventory:
        raise ValueError(f'Missing data for Aalop product {name}')
m = gp.Model('Aalop_Fulfillment')
x_vars = m.addVars(aalop_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[name] * x_vars[name] for name in aalop_products)), GRB.MAXIMIZE)
for name in aalop_products:
    m.addConstr(x_vars[name] <= initial_inventory[name], name=f'inv_{name}')
    m.addConstr(x_vars[name] <= demand[name], name=f'dem_{name}')
    m.addConstr(x_vars[name] >= 0, name=f'nonneg_{name}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')