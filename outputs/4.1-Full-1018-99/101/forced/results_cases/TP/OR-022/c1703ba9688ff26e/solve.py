LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv","values":{"Product Name":"27in 4K Gaming Monitor","Revenue":"261.2933","Demand":"12474","Initial Inventory":"62440"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv","values":{"Product Name":"27in FHD Monitor","Revenue":"52.4965","Demand":"15057","Initial Inventory":"75500"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '52.4965', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        inventory[pname] = int(vals['Initial Inventory'])
    except (KeyError, ValueError):
        raise ValueError(f'Missing or invalid data for product {pname}')
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('27in_Product_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= inventory[p] for p in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')