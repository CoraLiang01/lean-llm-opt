LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv","values":{"Product Name":"Aalopuri","Revenue":"20","Demand":"1483","Initial Inventory":"10440.0"}}]'
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
    if vals.get('Product Name', '').startswith('Aalop'):
        product = vals['Product Name']
        aalop_products.append(product)
        try:
            revenue[product] = float(vals['Revenue'])
            demand[product] = int(float(vals['Demand']))
            initial_inventory[product] = float(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {product}: {e}')
for product in aalop_products:
    if product not in revenue or product not in demand or product not in initial_inventory:
        raise ValueError(f'Missing data for product {product}')
m = gp.Model('Aalop_Inventory_Optimization')
x_vars = m.addVars(aalop_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in aalop_products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= initial_inventory[i] for i in aalop_products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in aalop_products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')