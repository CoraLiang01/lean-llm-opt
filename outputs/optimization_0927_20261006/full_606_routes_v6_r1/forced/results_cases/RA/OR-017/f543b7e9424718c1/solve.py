LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZ2AO",\n      "Revenue": "24.38",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZDW7",\n      "Revenue": "30.12",\n      "Demand": "4",\n      "Initial Inventory": "20.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZM1A",\n      "Revenue": "19.52",\n      "Demand": "82",\n      "Initial Inventory": "530.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZNC5",\n      "Revenue": "10.79",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZX6K",\n      "Revenue": "111.81",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZ2AO', 'Revenue': '24.38', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZDW7', 'Revenue': '30.12', 'Demand': '4', 'Initial Inventory': '20.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZM1A', 'Revenue': '19.52', 'Demand': '82', 'Initial Inventory': '530.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZNC5', 'Revenue': '10.79', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZX6K', 'Revenue': '111.81', 'Demand': '2', 'Initial Inventory': '10.0'}}]
import gurobipy as gp
from gurobipy import GRB
sku_data = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    sku = vals['SKU']
    revenue = float(vals['Revenue'])
    demand = int(float(vals['Demand']))
    inventory = int(float(vals['Initial Inventory']))
    sku_data[sku] = {'Revenue': revenue, 'Demand': demand, 'Inventory': inventory}
skus = list(sku_data.keys())
for sku in skus:
    if not all((k in sku_data[sku] for k in ['Revenue', 'Demand', 'Inventory'])):
        raise ValueError(f'Missing data for SKU {sku}')
m = gp.Model('RetailStore_ZZ_Fulfillment')
x_vars = m.addVars(skus, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((sku_data[sku]['Revenue'] * x_vars[sku] for sku in skus)), GRB.MAXIMIZE)
for sku in skus:
    m.addConstr(x_vars[sku] <= sku_data[sku]['Demand'], name=f'demand_{sku}')
    m.addConstr(x_vars[sku] <= sku_data[sku]['Inventory'], name=f'inv_{sku}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')