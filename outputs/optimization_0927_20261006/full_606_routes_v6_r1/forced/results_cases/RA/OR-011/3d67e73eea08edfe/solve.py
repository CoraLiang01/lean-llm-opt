LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv", "values": {"id_number": "id999", "Revenue": "434.74", "Demand": "8171", "Initial Inventory": "56450"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
record = next((r for r in LEGACY_RECORDS if r['values'].get('id_number') == 'id999'))
revenue = float(record['values']['Revenue'])
demand = int(record['values']['Demand'])
initial_inventory = int(record['values']['Initial Inventory'])
m = gp.Model('id999_fulfillment')
x_vars = m.addVar(lb=0, ub=min(demand, initial_inventory), vtype=GRB.INTEGER, name='x')
m.setObjective(revenue * x_vars, GRB.MAXIMIZE)
m.addConstr(x_vars <= demand, name='demand')
m.addConstr(x_vars <= initial_inventory, name='inventory')
m.addConstr(x_vars >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')