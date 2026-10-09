LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Fruits", "Revenue": "60.8", "Demand": "678906", "Initial Inventory": "5034020.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Staples", "Revenue": "918.45", "Demand": "749927", "Initial Inventory": "5589290.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Vegetables", "Revenue": "77.52", "Demand": "699808", "Initial Inventory": "5202710.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
categories = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    cat = vals['Sub Category']
    categories.append(cat)
    try:
        revenue[cat] = float(vals['Revenue'])
        demand[cat] = int(float(vals['Demand']))
        inventory[cat] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Missing or invalid data for category '{cat}': {e}")
for cat in categories:
    if cat not in revenue or cat not in demand or cat not in inventory:
        raise ValueError(f"Missing data for category '{cat}'")
x_vars = {}
m = gp.Model('Organ_Product_Fulfillment')
for cat in categories:
    ub = min(demand[cat], inventory[cat])
    x_vars[cat] = m.addVar(lb=0, ub=ub, vtype=GRB.INTEGER, name=f'x_{categories.index(cat) + 1}')
m.setObjective(gp.quicksum((revenue[cat] * x_vars[cat] for cat in categories)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for cat in categories:
        print(f'{x_vars[cat].VarName}: {x_vars[cat].X}')
else:
    print(f'Solver status: {m.Status}')