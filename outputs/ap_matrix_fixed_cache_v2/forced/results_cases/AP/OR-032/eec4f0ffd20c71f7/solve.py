LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv",\n    "values": {\n      "Product_Name": "Books_15.15",\n      "Revenue": "15.15",\n      "Demand": "1980",\n      "Initial Inventory": "9920.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv",\n    "values": {\n      "Product_Name": "Books_30.3",\n      "Revenue": "30.3",\n      "Demand": "3024",\n      "Initial Inventory": "20160.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv",\n    "values": {\n      "Product_Name": "Books_45.45",\n      "Revenue": "45.45",\n      "Demand": "4536",\n      "Initial Inventory": "30000.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv",\n    "values": {\n      "Product_Name": "Books_60.6",\n      "Revenue": "60.6",\n      "Demand": "5601",\n      "Initial Inventory": "38360.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv",\n    "values": {\n      "Product_Name": "Books_75.75",\n      "Revenue": "75.75",\n      "Demand": "7567",\n      "Initial Inventory": "51450.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_15.15', 'Revenue': '15.15', 'Demand': '1980', 'Initial Inventory': '9920.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_30.3', 'Revenue': '30.3', 'Demand': '3024', 'Initial Inventory': '20160.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_45.45', 'Revenue': '45.45', 'Demand': '4536', 'Initial Inventory': '30000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_60.6', 'Revenue': '60.6', 'Demand': '5601', 'Initial Inventory': '38360.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_75.75', 'Revenue': '75.75', 'Demand': '7567', 'Initial Inventory': '51450.0'}}]
import gurobipy as gp
from gurobipy import GRB
books_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product_Name', '').startswith('Books_')]
books_ids = []
revenue = {}
demand = {}
init_inventory = {}
capacity = {}
for rec in books_records:
    name = rec['values']['Product_Name']
    books_ids.append(name)
    revenue[name] = float(rec['values']['Revenue'])
    demand[name] = int(float(rec['values']['Demand']))
    init_inventory[name] = float(rec['values']['Initial Inventory'])
    capacity[name] = min(init_inventory[name], demand[name])
for name in books_ids:
    if name not in revenue or name not in demand or name not in init_inventory:
        raise ValueError(f'Missing data for product {name}')
m = gp.Model('Books_Revenue_Maximization')
x_vars = m.addVars(books_ids, vtype=GRB.INTEGER, lb=0, ub=[capacity[name] for name in books_ids], name='')
m.setObjective(gp.quicksum((revenue[name] * x_vars[name] for name in books_ids)), GRB.MAXIMIZE)
for name in books_ids:
    m.addConstr(x_vars[name] <= capacity[name], name=f'cap_{name}')
    m.addConstr(x_vars[name] >= 0, name=f'nonneg_{name}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')