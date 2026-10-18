LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv","values":{"Product Name":"Aalopuri","Revenue":"20","Demand":"1483","Initial Inventory":"10440.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
aalopuri_record = None
for rec in records:
    if rec.get('values', {}).get('Product Name', '') == 'Aalopuri':
        aalopuri_record = rec['values']
        break
if aalopuri_record is None:
    raise ValueError('Aalopuri record not found in LEGACY_RECORDS.')
try:
    revenue = float(aalopuri_record['Revenue'])
    demand = int(float(aalopuri_record['Demand']))
    initial_inventory = int(float(aalopuri_record['Initial Inventory']))
except Exception as e:
    raise ValueError(f'Error parsing numeric fields: {e}')
m = gp.Model('Aalopuri_Fulfillment')
x_vars = m.addVar(lb=0, ub=min(initial_inventory, demand), vtype=GRB.INTEGER, name='x_Aalopuri')
m.setObjective(revenue * x_vars, GRB.MAXIMIZE)
m.addConstr(x_vars <= initial_inventory, name='inv')
m.addConstr(x_vars <= demand, name='dem')
m.addConstr(x_vars >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')