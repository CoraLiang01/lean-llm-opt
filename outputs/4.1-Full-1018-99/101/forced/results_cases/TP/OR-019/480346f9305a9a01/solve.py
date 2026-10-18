LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in 4K Gaming Monitor",\n      "Revenue": "389.99",\n      "Demand": "12474",\n      "Initial Inventory": "62440"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv",\n    "values": {\n      "Product Name": "27in FHD Monitor",\n      "Revenue": "149.99",\n      "Demand": "15057",\n      "Initial Inventory": "75500"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
initial_inventory = {}
capacity = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    if pname.startswith('27in'):
        products.append(pname)
        try:
            revenue[pname] = float(vals['Revenue'])
            demand[pname] = int(vals['Demand'])
            initial_inventory[pname] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pname}: {e}')
for pname in products:
    capacity[pname] = min(demand[pname], initial_inventory[pname])
if set(revenue) != set(products) or set(demand) != set(products) or set(initial_inventory) != set(products):
    raise ValueError('Missing data for some products in LEGACY_RECORDS.')
m = gp.Model('27in_Product_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, ub=[capacity[p] for p in products], vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] <= capacity[p], name=f'cap_{p}')
    m.addConstr(x_vars[p] >= 0, name=f'nonneg_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')