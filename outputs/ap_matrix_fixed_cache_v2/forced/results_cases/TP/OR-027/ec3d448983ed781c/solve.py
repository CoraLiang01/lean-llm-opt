LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Fruits", "Revenue": "60.8", "Demand": "678906", "Initial Inventory": "5034020.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Staples", "Revenue": "918.45", "Demand": "749927", "Initial Inventory": "5589290.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Vegetables", "Revenue": "77.52", "Demand": "699808", "Initial Inventory": "5202710.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
organ_products = []
for rec in LEGACY_RECORDS:
    if rec['values'].get('Sub Category', '').startswith('Organic '):
        organ_products.append(rec)
I = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in organ_products:
    subcat = rec['values']['Sub Category']
    I.append(subcat)
    try:
        revenue[subcat] = float(rec['values']['Revenue'])
        demand[subcat] = float(rec['values']['Demand'])
        initial_inventory[subcat] = float(rec['values']['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid numeric data for {subcat}: {e}')
for subcat in I:
    if subcat not in revenue or subcat not in demand or subcat not in initial_inventory:
        raise ValueError(f'Missing data for {subcat}')
m = gp.Model('Supermart_Organ_Revenue')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in I), name='')
m.addConstrs((x_vars[i] <= initial_inventory[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')