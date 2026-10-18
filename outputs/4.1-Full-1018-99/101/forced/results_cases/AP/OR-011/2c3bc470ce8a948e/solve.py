LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv",\n    "values": {\n      "id_number": "id999",\n      "Revenue": "434.74",\n      "Demand": "8171",\n      "Initial Inventory": "56450"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
product_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('id_number') == 'id999']
if not product_records:
    raise ValueError("No product record with id_number 'id999' found in LEGACY_RECORDS.")
rec = product_records[0]['values']
id_number = rec['id_number']
try:
    revenue = float(rec['Revenue'])
    demand = int(float(rec['Demand']))
    inventory = int(float(rec['Initial Inventory']))
except Exception as e:
    raise ValueError(f'Error parsing numeric fields: {e}')
m = gp.Model('Supermarket_Inventory_Fulfillment')
x1_vars = m.addVar(vtype=GRB.INTEGER, lb=0, ub=min(demand, inventory), name='x1')
m.setObjective(revenue * x1_vars, GRB.MAXIMIZE)
m.addConstr(x1_vars <= demand, name='demand')
m.addConstr(x1_vars <= inventory, name='inventory')
m.addConstr(x1_vars >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')