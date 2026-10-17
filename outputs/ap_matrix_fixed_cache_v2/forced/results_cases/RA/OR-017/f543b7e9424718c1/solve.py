LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZ2AO",\n      "Revenue": "24.38",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZDW7",\n      "Revenue": "30.12",\n      "Demand": "4",\n      "Initial Inventory": "20.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZM1A",\n      "Revenue": "19.52",\n      "Demand": "82",\n      "Initial Inventory": "530.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZNC5",\n      "Revenue": "10.79",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZX6K",\n      "Revenue": "111.81",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZ2AO', 'Revenue': '24.38', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZDW7', 'Revenue': '30.12', 'Demand': '4', 'Initial Inventory': '20.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZM1A', 'Revenue': '19.52', 'Demand': '82', 'Initial Inventory': '530.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZNC5', 'Revenue': '10.79', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZX6K', 'Revenue': '111.81', 'Demand': '2', 'Initial Inventory': '10.0'}}]
import gurobipy as gp
from gurobipy import GRB
zz_skus = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    sku = vals['SKU']
    if sku.startswith('ZZ'):
        zz_skus.append(sku)
        try:
            revenue[sku] = float(vals['Revenue'])
            demand[sku] = int(float(vals['Demand']))
            initial_inventory[sku] = int(float(vals['Initial Inventory']))
        except Exception as e:
            raise ValueError(f'Invalid data for SKU {sku}: {e}')
for sku in zz_skus:
    if sku not in revenue or sku not in demand or sku not in initial_inventory:
        raise ValueError(f'Missing data for SKU {sku}')
upper_bounds = {sku: min(demand[sku], initial_inventory[sku]) for sku in zz_skus}
m = gp.Model('RetailStore_ZZ_Revenue')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(zz_skus, lb=0, ub=[upper_bounds[sku] for sku in zz_skus], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[sku] * x_vars[sku] for sku in zz_skus)), GRB.MAXIMIZE)
for sku in zz_skus:
    m.addConstr(x_vars[sku] >= 0, name=f'nonneg_{sku}')
    m.addConstr(x_vars[sku] <= upper_bounds[sku], name=f'cap_{sku}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')