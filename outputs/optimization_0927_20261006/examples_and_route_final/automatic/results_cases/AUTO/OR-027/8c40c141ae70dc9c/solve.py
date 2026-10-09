LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv","values":{"Sub Category":"Organic Fruits","Revenue":"60.8","Demand":"678906","Initial Inventory":"5034020.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv","values":{"Sub Category":"Organic Staples","Revenue":"918.45","Demand":"749927","Initial Inventory":"5589290.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv","values":{"Sub Category":"Organic Vegetables","Revenue":"77.52","Demand":"699808","Initial Inventory":"5202710.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
organ_products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    subcat = vals['Sub Category']
    organ_products.append(subcat)
    try:
        revenue[subcat] = float(vals['Revenue'])
        demand[subcat] = int(float(vals['Demand']))
        initial_inventory[subcat] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for {subcat}: {e}')
for subcat in organ_products:
    if subcat not in revenue or subcat not in demand or subcat not in initial_inventory:
        raise ValueError(f'Missing data for {subcat}')
upper_bounds = {subcat: min(initial_inventory[subcat], demand[subcat]) for subcat in organ_products}
m = gp.Model('Organ_Product_Fulfillment')
x_vars = m.addVars(organ_products, lb=0, ub=[upper_bounds[subcat] for subcat in organ_products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[subcat] * x_vars[subcat] for subcat in organ_products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')