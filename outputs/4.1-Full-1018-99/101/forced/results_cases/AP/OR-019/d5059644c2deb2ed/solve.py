LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in 4K Gaming Monitor",\n      "Revenue": "389.99",\n      "Demand": "12474",\n      "Initial Inventory": "62440"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in FHD Monitor",\n      "Revenue": "149.99",\n      "Demand": "15057",\n      "Initial Inventory": "75500"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenues = {}
demands = {}
inventories = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    products.append(pname)
    revenues[pname] = float(vals['Revenue'])
    demands[pname] = int(vals['Demand'])
    inventories[pname] = int(vals['Initial Inventory'])
for pname in products:
    if pname not in revenues or pname not in demands or pname not in inventories:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Supermarket_27in_Revenue')
x_vars = m.addVars(products, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenues[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] >= 0, name=f'nonneg_{p}')
    m.addConstr(x_vars[p] <= min(inventories[p], demands[p]), name=f'cap_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')