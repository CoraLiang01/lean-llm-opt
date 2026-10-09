LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv", "values": {"Product Name": "FDK57", "Revenue": "119.144", "Demand": "30", "Initial Inventory": "200"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv", "values": {"Product Name": "FDK57", "Revenue": "119.144", "Demand": "40", "Initial Inventory": "100"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv", "values": {"Product Name": "FDK57", "Revenue": "120.144", "Demand": "50", "Initial Inventory": "150"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv", "values": {"Product Name": "FDK57", "Revenue": "121.244", "Demand": "30", "Initial Inventory": "200"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv", "values": {"Product Name": "FDK57", "Revenue": "120.544", "Demand": "10", "Initial Inventory": "150"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv", "values": {"Product Name": "FDK57", "Revenue": "121.244", "Demand": "30", "Initial Inventory": "150"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '120.544', 'Demand': '10', 'Initial Inventory': '150'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '150'}}]
import gurobipy as gp
from gurobipy import GRB
fdk57_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name') == 'FDK57']
revenues = []
demands = []
inventories = []
for (idx, rec) in enumerate(fdk57_records):
    vals = rec['values']
    try:
        revenue = float(vals['Revenue'])
        demand = int(vals['Demand'])
        inventory = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} in record {idx + 1}')
    except Exception as e:
        raise ValueError(f'Invalid data in record {idx + 1}: {e}')
    revenues.append(revenue)
    demands.append(demand)
    inventories.append(inventory)
n = len(fdk57_records)
if n != 6:
    raise ValueError(f'Expected 6 FDK57 records, found {n}')
m = gp.Model('FDK57_Allocation')
x_vars = m.addVars(range(n), lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenues[i] * x_vars[i] for i in range(n))), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demands[i] for i in range(n)), name='')
m.addConstrs((x_vars[i] <= inventories[i] for i in range(n)), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')