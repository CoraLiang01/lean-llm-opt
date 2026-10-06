LEGACY_OBSERVATION = 'id_number,Revenue,Initial Inventory,Demand\nid999,434.74,56450,8171'
LEGACY_RECORDS = [{'source': '', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Initial Inventory': '56450', 'Demand': '8171'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
for rec in records:
    vals = rec['values']
    if vals.get('id_number') == 'id999':
        product_id = vals['id_number']
        try:
            revenue = float(vals['Revenue'])
            inventory = int(float(vals['Initial Inventory']))
            demand = int(float(vals['Demand']))
        except Exception as e:
            raise ValueError(f'Invalid data in LEGACY_RECORDS for id999: {e}')
        break
else:
    raise ValueError('No record found for id999 in LEGACY_RECORDS')
if not isinstance(revenue, float) or not isinstance(inventory, int) or (not isinstance(demand, int)):
    raise ValueError('Coefficient types are incorrect for id999.')
m = gp.Model('id999_fulfillment')
x = m.addVar(lb=0, ub=min(inventory, demand), vtype=GRB.INTEGER, name='x')
m.setObjective(revenue * x, GRB.MAXIMIZE)
m.addConstr(x <= inventory, name='inv')
m.addConstr(x <= demand, name='dem')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')