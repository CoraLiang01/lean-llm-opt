LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv", "values": {"Product Name": "Baby Food_255.28", "Revenue": "255.28", "Demand": "3066513", "Initial Inventory": "22749210"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '3066513', 'Initial Inventory': '22749210'}}]
import gurobipy as gp
from gurobipy import GRB
product_records = [rec for rec in LEGACY_RECORDS if rec['source']]
if not product_records:
    raise ValueError('No product records found with a source.')
products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in product_records:
    vals = rec['values']
    pname = vals['Product Name']
    try:
        r = float(vals['Revenue'])
        d = int(vals['Demand'])
        s = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pname}: {e}')
    products.append(pname)
    revenue[pname] = r
    demand[pname] = d
    initial_inventory[pname] = s
m = gp.Model('Baby_Product_Fulfillment')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x_vars[p] <= initial_inventory[p], name=f'inventory_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')