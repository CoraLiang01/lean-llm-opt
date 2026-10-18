LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in 4K Gaming Monitor",\n      "Revenue": "389.99",\n      "Demand": "12474",\n      "Initial Inventory": "62440"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in FHD Monitor",\n      "Revenue": "149.99",\n      "Demand": "15057",\n      "Initial Inventory": "75500"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
initial_inventory = {}
for record in LEGACY_RECORDS:
    vals = record['values']
    pname = vals['Product Name']
    products.append(pname)
    revenue[pname] = float(vals['Revenue'])
    demand[pname] = int(vals['Demand'])
    initial_inventory[pname] = int(vals['Initial Inventory'])
m = gp.Model('27in_Product_Fulfillment')
x_vars = m.addVars(products, lb=0, ub={p: min(demand[p], initial_inventory[p]) for p in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')