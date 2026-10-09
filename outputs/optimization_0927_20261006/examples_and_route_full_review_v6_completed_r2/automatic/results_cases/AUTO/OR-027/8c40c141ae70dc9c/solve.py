LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv",\n    "values": {\n      "Sub Category": "Organic Fruits",\n      "Revenue": "60.8",\n      "Demand": "678906",\n      "Initial Inventory": "5034020.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv",\n    "values": {\n      "Sub Category": "Organic Staples",\n      "Revenue": "918.45",\n      "Demand": "749927",\n      "Initial Inventory": "5589290.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv",\n    "values": {\n      "Sub Category": "Organic Vegetables",\n      "Revenue": "77.52",\n      "Demand": "699808",\n      "Initial Inventory": "5202710.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
organ_products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    subcat = vals['Sub Category']
    organ_products.append(subcat)
    try:
        revenue[subcat] = float(vals['Revenue'])
        demand[subcat] = int(float(vals['Demand']))
        inventory[subcat] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for {subcat}: {e}')
for p in organ_products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Organ_Product_Fulfillment')
x_vars = m.addVars(organ_products, lb=0, ub={p: min(demand[p], inventory[p]) for p in organ_products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in organ_products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')