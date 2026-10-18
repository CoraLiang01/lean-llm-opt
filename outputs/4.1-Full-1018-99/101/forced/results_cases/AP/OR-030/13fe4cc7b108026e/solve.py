LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "40",\n      "Initial Inventory": "100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.144",\n      "Demand": "50",\n      "Initial Inventory": "150"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "121.244",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.544",\n      "Demand": "10",\n      "Initial Inventory": "150"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.544', 'Demand': '10', 'Initial Inventory': '150'}}]
import gurobipy as gp
from gurobipy import GRB
fdk57_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name') == 'FDK57']
if len(fdk57_records) != 5:
    raise ValueError('Expected 5 FDK57 records, found {}'.format(len(fdk57_records)))
indices = list(range(1, len(fdk57_records) + 1))
revenue = {}
inventory = {}
demand = {}
for idx, rec in zip(indices, fdk57_records):
    vals = rec['values']
    try:
        revenue[idx] = float(vals['Revenue'])
        inventory[idx] = int(vals['Initial Inventory'])
        demand[idx] = int(vals['Demand'])
    except Exception as e:
        raise ValueError(f'Error parsing record {idx}: {e}')
upper_bounds = {i: min(inventory[i], demand[i]) for i in indices}
m = gp.Model('FDK57_RevMax')
x_vars = m.addVars(indices, vtype=GRB.INTEGER, lb=0, ub=[upper_bounds[i] for i in indices], name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in indices)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')