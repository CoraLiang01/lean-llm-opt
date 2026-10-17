LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "119.144",\n      "Demand": "40",\n      "Initial Inventory": "100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "120.144",\n      "Demand": "50",\n      "Initial Inventory": "150"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv",\n    "values": {\n      "Product Name": "FDK57",\n      "Revenue": "121.244",\n      "Demand": "30",\n      "Initial Inventory": "200"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '200'}}]
import gurobipy as gp
from gurobipy import GRB
fdk57_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name') == 'FDK57']
model_keys = list(range(1, len(fdk57_records) + 1))
revenue = {}
demand = {}
initial_inventory = {}
capacity = {}
for idx, rec in enumerate(fdk57_records, 1):
    vals = rec['values']
    try:
        revenue[idx] = float(vals['Revenue'])
        demand[idx] = float(vals['Demand'])
        initial_inventory[idx] = float(vals['Initial Inventory'])
        capacity[idx] = min(demand[idx], initial_inventory[idx])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} in record {idx}')
    except Exception as e:
        raise ValueError(f'Error parsing record {idx}: {e}')
if not set(revenue) == set(demand) == set(initial_inventory) == set(capacity) == set(model_keys):
    raise ValueError('Parameter keys mismatch in FDK57 records.')
m = gp.Model('FDK57_Car_Dealership')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(model_keys, lb=0, ub=capacity, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in model_keys)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in model_keys:
        print(f'x[{i}]: {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')