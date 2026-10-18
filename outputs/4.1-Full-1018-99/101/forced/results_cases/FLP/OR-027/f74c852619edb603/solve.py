LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv",\n    "values": {\n      "Sub Category": "Organic Fruits",\n      "Revenue": "60.8",\n      "Demand": "678906",\n      "Initial Inventory": "5034020.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv",\n    "values": {\n      "Sub Category": "Organic Staples",\n      "Revenue": "918.45",\n      "Demand": "749927",\n      "Initial Inventory": "5589290.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv",\n    "values": {\n      "Sub Category": "Organic Vegetables",\n      "Revenue": "77.52",\n      "Demand": "699808",\n      "Initial Inventory": "5202710.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    prod = vals['Sub Category']
    products.append(prod)
    try:
        revenue[prod] = float(vals['Revenue'])
        demand[prod] = int(float(vals['Demand']))
        initial_inventory[prod] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f"Missing or invalid data for product '{prod}': {e}")
for prod in products:
    if prod not in revenue or prod not in demand or prod not in initial_inventory:
        raise ValueError(f"Missing data for product '{prod}'")
upper_bounds = {prod: min(initial_inventory[prod], demand[prod]) for prod in products}
m = gp.Model('Supermarket_Organ_Revenue')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, ub=[upper_bounds[prod] for prod in products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[prod] * x_vars[prod] for prod in products)), GRB.MAXIMIZE)
for prod in products:
    m.addConstr(x_vars[prod] >= 0, name=f'lb_{prod}')
    m.addConstr(x_vars[prod] <= upper_bounds[prod], name=f'ub_{prod}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')