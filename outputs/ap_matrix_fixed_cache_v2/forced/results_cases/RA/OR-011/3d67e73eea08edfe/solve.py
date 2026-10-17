LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv", "values": {"id_number": "id999", "Revenue": "434.74", "Demand": "8171", "Initial Inventory": "56450"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
for rec in records:
    if rec['values'].get('id_number') == 'id999':
        product_id = rec['values']['id_number']
        revenue = float(rec['values']['Revenue'])
        demand = int(rec['values']['Demand'])
        inventory = int(rec['values']['Initial Inventory'])
        break
else:
    raise ValueError('No record found for id999')
m = gp.Model('id999_fulfillment')
x_vars = m.addVar(lb=0, vtype=GRB.INTEGER, name='x')
m.setObjective(revenue * x_vars, GRB.MAXIMIZE)
m.addConstr(x_vars <= demand, name='demand')
m.addConstr(x_vars <= inventory, name='inventory')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')