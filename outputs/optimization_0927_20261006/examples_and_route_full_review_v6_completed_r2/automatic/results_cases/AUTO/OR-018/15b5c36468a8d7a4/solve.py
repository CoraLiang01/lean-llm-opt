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
    if 'Baby' in vals.get('Product Name', ''):
        product = vals['Product Name']
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
m = gp.Model('Baby_Product_Fulfillment')
x_vars = m.addVars(baby_products, lb=0, ub={p: min(demand[p], initial_inventory[p]) for p in baby_products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in baby_products)), GRB.MAXIMIZE)
for p in baby_products:
    m.addConstr(x_vars[p] <= initial_inventory[p], name=f'inventory_{p}')
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x_vars[p] >= 0, name=f'nonneg_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')