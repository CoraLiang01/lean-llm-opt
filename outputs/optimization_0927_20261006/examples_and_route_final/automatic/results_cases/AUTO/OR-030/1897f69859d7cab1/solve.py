LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "40",\n      "Initial Inventory": "100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "121.244",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.144",\n      "Demand": "50",\n      "Initial Inventory": "150"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.844",\n      "Demand": "50",\n      "Initial Inventory": "150"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.844', 'Demand': '50', 'Initial Inventory': '150'}}]
import gurobipy as gp
from gurobipy import GRB
fdk57_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name') == 'FDK57']
if len(fdk57_records) != 5:
    raise ValueError('Expected 5 FDK57 records, got {}'.format(len(fdk57_records)))
revenues = []
demands = []
inventories = []
for rec in fdk57_records:
    vals = rec['values']
    try:
        revenues.append(float(vals['Revenue']))
        demands.append(int(vals['Demand']))
        inventories.append(int(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Error parsing record {vals}: {e}')
n = len(fdk57_records)
indices = list(range(n))
upper_bounds = [min(demands[i], inventories[i]) for i in indices]
m = gp.Model('FDK57_Allocation')
x_vars = m.addVars(indices, lb=0, ub={i: upper_bounds[i] for i in indices}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenues[i] * x_vars[i] for i in indices)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in indices:
        print(f'x[{i}]: {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')