LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in 4K Gaming Monitor",\n      "Revenue": "389.99",\n      "Demand": "12474",\n      "Initial Inventory": "62440"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in FHD Monitor",\n      "Revenue": "149.99",\n      "Demand": "15057",\n      "Initial Inventory": "75500"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products_27in = []
for rec in LEGACY_RECORDS:
    name = rec['values'].get('Product Name', '')
    if name.startswith('27in'):
        products_27in.append(name)
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    name = rec['values'].get('Product Name', '')
    if name in products_27in:
        try:
            revenue[name] = float(rec['values']['Revenue'])
            demand[name] = int(rec['values']['Demand'])
            inventory[name] = int(rec['values']['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Missing or invalid data for product '{name}': {e}")
for name in products_27in:
    if name not in revenue or name not in demand or name not in inventory:
        raise ValueError(f"Missing data for product '{name}'")
upper_bounds = {name: min(demand[name], inventory[name]) for name in products_27in}
m = gp.Model('27in_Product_Fulfillment')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products_27in, lb=0, ub=[upper_bounds[name] for name in products_27in], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[name] * x_vars[name] for name in products_27in)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')