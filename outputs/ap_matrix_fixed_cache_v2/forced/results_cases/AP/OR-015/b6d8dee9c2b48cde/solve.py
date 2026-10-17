LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv",\n    "values": {\n      "Product Name": "Aalopuri",\n      "Revenue": "20",\n      "Demand": "1483",\n      "Initial Inventory": "10440.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
aalopuri_data = None
for rec in records:
    if rec['values'].get('Product Name') == 'Aalopuri':
        aalopuri_data = rec['values']
        break
if aalopuri_data is None:
    raise ValueError("No data found for 'Aalopuri' in LEGACY_RECORDS.")
try:
    revenue = float(aalopuri_data['Revenue'])
    demand = int(float(aalopuri_data['Demand']))
    initial_inventory = int(float(aalopuri_data['Initial Inventory']))
except Exception as e:
    raise ValueError(f'Error parsing numeric fields: {e}')
m = gp.Model('Aalopuri_Revenue_Max')
x_vars = m.addVar(vtype=GRB.INTEGER, lb=0, ub=min(demand, initial_inventory), name='x_Aalopuri')
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