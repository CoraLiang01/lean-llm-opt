LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "40",\n      "Initial Inventory": "100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.144",\n      "Demand": "50",\n      "Initial Inventory": "150"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "121.244",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.544",\n      "Demand": "10",\n      "Initial Inventory": "150"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.544', 'Demand': '10', 'Initial Inventory': '150'}}]
import gurobipy as gp
from gurobipy import GRB
fdk57_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name') == 'FDK57']
revenues = []
demands = []
inventories = []
for idx, rec in enumerate(fdk57_records):
    vals = rec['values']
    try:
        revenues.append(float(vals['Revenue']))
        demands.append(int(vals['Demand']))
        inventories.append(int(vals['Initial Inventory']))
    except KeyError as e:
        raise ValueError(f'Missing required field {e} in record {idx}')
n = len(fdk57_records)
if not len(revenues) == len(demands) == len(inventories) == n:
    raise ValueError('Data dimension mismatch in LEGACY_RECORDS for FDK57.')
upper_bounds = [min(demands[i], inventories[i]) for i in range(n)]
m = gp.Model('FDK57_Car_Dealership_Allocation')
x_vars = m.addVars(n, lb=0, ub=upper_bounds, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenues[i] * x_vars[i] for i in range(n))), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in range(n):
        print(f'x[{i + 1}]: {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')