LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv",\n    "values": {\n      "Product Name": "Baby Food_255.28",\n      "Revenue": "255.28",\n      "Demand": "3066513",\n      "Initial Inventory": "22749210"\n    }\n  }\n]'
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
    pname = vals['Product Name']
    baby_products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        initial_inventory[pname] = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} for product {pname}')
for pname in baby_products:
    if pname not in revenue or pname not in demand or pname not in initial_inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Baby_Product_Fulfillment')
x_vars = m.addVars(baby_products, lb=0, ub=[demand[p] for p in baby_products], vtype=GRB.INTEGER, name='')
for p in baby_products:
    m.addConstr(x_vars[p] <= initial_inventory[p], name=f'inv_{p}')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in baby_products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')