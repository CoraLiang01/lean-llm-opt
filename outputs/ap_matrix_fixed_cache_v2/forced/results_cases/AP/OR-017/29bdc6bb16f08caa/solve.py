LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZ2AO",\n      "Revenue": "24.38",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZDW7",\n      "Revenue": "30.12",\n      "Demand": "4",\n      "Initial Inventory": "20.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZM1A",\n      "Revenue": "19.52",\n      "Demand": "82",\n      "Initial Inventory": "530.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZNC5",\n      "Revenue": "10.79",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv",\n    "values": {\n      "SKU": "ZZX6K",\n      "Revenue": "111.81",\n      "Demand": "2",\n      "Initial Inventory": "10.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZ2AO', 'Revenue': '24.38', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZDW7', 'Revenue': '30.12', 'Demand': '4', 'Initial Inventory': '20.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZM1A', 'Revenue': '19.52', 'Demand': '82', 'Initial Inventory': '530.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZNC5', 'Revenue': '10.79', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv', 'values': {'SKU': 'ZZX6K', 'Revenue': '111.81', 'Demand': '2', 'Initial Inventory': '10.0'}}]
import gurobipy as gp
from gurobipy import GRB
zz_products = []
for rec in LEGACY_RECORDS:
    vals = rec['values']
    sku = vals['SKU']
    if sku.startswith('ZZ'):
        zz_products.append({'SKU': sku, 'Revenue': float(vals['Revenue']), 'Demand': int(float(vals['Demand'])), 'Initial Inventory': int(float(vals['Initial Inventory']))})
sku_set = set()
for prod in zz_products:
    if prod['SKU'] in sku_set:
        raise ValueError(f"Duplicate SKU: {prod['SKU']}")
    sku_set.add(prod['SKU'])
    for field in ['Revenue', 'Demand', 'Initial Inventory']:
        if field not in prod:
            raise ValueError(f"Missing {field} for SKU {prod['SKU']}")
skus = [prod['SKU'] for prod in zz_products]
revenue = {prod['SKU']: prod['Revenue'] for prod in zz_products}
demand = {prod['SKU']: prod['Demand'] for prod in zz_products}
init_inventory = {prod['SKU']: prod['Initial Inventory'] for prod in zz_products}
m = gp.Model('RetailStore_ZZ_MaxRevenue')
x_vars = m.addVars(skus, vtype=GRB.INTEGER, lb=0, name='')
for sku in skus:
    m.addConstr(x_vars[sku] <= init_inventory[sku], name=f'inv_{sku}')
    m.addConstr(x_vars[sku] <= demand[sku], name=f'dem_{sku}')
m.setObjective(gp.quicksum((revenue[sku] * x_vars[sku] for sku in skus)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')