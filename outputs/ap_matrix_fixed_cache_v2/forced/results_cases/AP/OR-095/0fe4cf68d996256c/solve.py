LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv",\n    "values": {\n      "Resource": "LaborHours",\n      "MonthlyLimit": "5000"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv",\n    "values": {\n      "Resource": "MaterialA",\n      "MonthlyLimit": "24000"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv",\n    "values": {\n      "Resource": "MaterialB",\n      "MonthlyLimit": "15000"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv', 'values': {'Resource': 'LaborHours', 'MonthlyLimit': '5000'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv', 'values': {'Resource': 'MaterialA', 'MonthlyLimit': '24000'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv', 'values': {'Resource': 'MaterialB', 'MonthlyLimit': '15000'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv', 'values': {'Resource': 'LaborHours', 'MonthlyLimit': '5000'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv', 'values': {'Resource': 'MaterialA', 'MonthlyLimit': '24000'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv', 'values': {'Resource': 'MaterialB', 'MonthlyLimit': '15000'}}]
resource_limits = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('resource_limits.csv'):
        resource = rec['values']['Resource']
        limit = float(rec['values']['MonthlyLimit'])
        resource_limits[resource] = limit
widget_data = [{'Product': 'Widget1', 'LaborHours': 1.6, 'MaterialA': 24, 'MaterialB': 14, 'Profit': 525}, {'Product': 'Widget2', 'LaborHours': 2, 'MaterialA': 20, 'MaterialB': 10, 'Profit': 678}, {'Product': 'Widget3', 'LaborHours': 2.5, 'MaterialA': 12, 'MaterialB': 18, 'Profit': 812}, {'Product': 'Widget4', 'LaborHours': 1.9, 'MaterialA': 21, 'MaterialB': 15, 'Profit': 769}, {'Product': 'Widget5', 'LaborHours': 0.0, 'MaterialA': 15, 'MaterialB': 26, 'Profit': 952}, {'Product': 'Widget6', 'LaborHours': 0.1, 'MaterialA': 24, 'MaterialB': 17, 'Profit': 987}, {'Product': 'Widget7', 'LaborHours': 1.2, 'MaterialA': 15, 'MaterialB': 30, 'Profit': 644}, {'Product': 'Widget8', 'LaborHours': 1.3, 'MaterialA': 21, 'MaterialB': 24, 'Profit': 795}, {'Product': 'Widget9', 'LaborHours': 0.4, 'MaterialA': 20, 'MaterialB': 30, 'Profit': 829}, {'Product': 'Widget10', 'LaborHours': 0.9, 'MaterialA': 18, 'MaterialB': 27, 'Profit': 574}, {'Product': 'Widget141', 'LaborHours': 1.2, 'MaterialA': 11, 'MaterialB': 16, 'Profit': 593}]
if len(widget_data) != 141:
    raise ValueError('widget_data must contain all 141 widgets as per the query.')
widget_ids = [i + 1 for i in range(len(widget_data))]
product_names = [w['Product'] for w in widget_data]
L = {i + 1: float(w['LaborHours']) for i, w in enumerate(widget_data)}
A = {i + 1: float(w['MaterialA']) for i, w in enumerate(widget_data)}
B = {i + 1: float(w['MaterialB']) for i, w in enumerate(widget_data)}
P = {i + 1: float(w['Profit']) for i, w in enumerate(widget_data)}
m = gp.Model('widget_production')
x_vars = m.addVars(widget_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
y_var = m.addVar(lb=0, ub=1500, vtype=GRB.CONTINUOUS, name='y')
z_var = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='z')
m.setObjective(gp.quicksum((P[i] * x_vars[i] for i in widget_ids)) + 300 * y_var - 200 * z_var, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((L[i] * x_vars[i] for i in widget_ids)) <= resource_limits['LaborHours'], name='labor')
m.addConstr(gp.quicksum((A[i] * x_vars[i] for i in widget_ids)) <= resource_limits['MaterialA'], name='matA')
m.addConstr(gp.quicksum((B[i] * x_vars[i] for i in widget_ids)) <= resource_limits['MaterialB'], name='matB')
m.addConstr(5 * x_vars[3] == y_var + z_var, name='catalyst_balance')
m.addConstr(y_var >= 0, name='y_nonneg')
m.addConstr(y_var <= 1500, name='y_cap')
m.addConstr(z_var >= 0, name='z_nonneg')
for i in widget_ids:
    m.addConstr(x_vars[i] >= 0, name=f'x_nonneg_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in widget_ids:
        print(f'x[{i}]: {x_vars[i].X}')
    print(f'y: {y_var.X}')
    print(f'z: {z_var.X}')
else:
    print(f'Solver status: {m.Status}')