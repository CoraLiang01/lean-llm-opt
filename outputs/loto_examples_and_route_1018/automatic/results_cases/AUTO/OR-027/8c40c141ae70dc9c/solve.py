LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Fruits", "Revenue": "60.8", "Demand": "678906", "Initial Inventory": "5034020.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Staples", "Revenue": "918.45", "Demand": "749927", "Initial Inventory": "5589290.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv", "values": {"Sub Category": "Organic Vegetables", "Revenue": "77.52", "Demand": "699808", "Initial Inventory": "5202710.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
subcats = []
revenue = {}
demand = {}
init_inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    subcat = vals['Sub Category']
    subcats.append(subcat)
    revenue[subcat] = float(vals['Revenue'])
    demand[subcat] = int(float(vals['Demand']))
    init_inventory[subcat] = int(float(vals['Initial Inventory']))
ub = {sc: min(demand[sc], init_inventory[sc]) for sc in subcats}
m = gp.Model('Organ_Product_Fulfillment')
x_vars = m.addVars(subcats, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[sc] * x_vars[sc] for sc in subcats)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')