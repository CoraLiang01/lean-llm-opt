LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "40",\n      "Initial Inventory": "100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "121.244",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.144",\n      "Demand": "50",\n      "Initial Inventory": "150"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.844",\n      "Demand": "50",\n      "Initial Inventory": "150"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.844', 'Demand': '50', 'Initial Inventory': '150'}}]
import gurobipy as gp
from gurobipy import GRB
rows = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name') == 'FDK57']
if len(rows) != 5:
    raise ValueError('Expected 5 FDK57 rows in LEGACY_RECORDS, found {}'.format(len(rows)))
revenues = []
demands = []
inventories = []
for idx, rec in enumerate(rows):
    vals = rec['values']
    try:
        revenues.append(float(vals['Revenue']))
        demands.append(int(vals['Demand']))
        inventories.append(int(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Error parsing row {idx}: {e}')
n = len(rows)
indices = list(range(n))
m = gp.Model('FDK57_RevMax')
x_vars = m.addVars(indices, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenues[k] * x_vars[k] for k in indices)), GRB.MAXIMIZE)
for k in indices:
    m.addConstr(x_vars[k] <= demands[k], name=f'demand_{k}')
    m.addConstr(x_vars[k] <= inventories[k], name=f'inventory_{k}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')