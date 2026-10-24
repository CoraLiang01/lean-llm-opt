CSVQA_DATA = {'query': 'The company operates in the European market and offers a variety of products with revenue data provided in the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'EuropeSalesRecords.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '765850', 'Initial Inventory': '5627060', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB
table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
I = []
params = {}
for rec in records:
    pname = rec['values']['Product Name']
    if 'Baby' in pname:
        i = rec['data_row_number']
        I.append(i)
        try:
            r = float(rec['values']['Revenue'])
        except Exception:
            r = 0.0
        try:
            d = int(rec['values']['Demand'])
        except Exception:
            d = 0
        try:
            s = int(rec['values']['Initial Inventory'])
        except Exception:
            s = 0
        params[i] = {'r': r, 'd': d, 's': s}
m = gurobipy.Model()
m.setParam('MIPGap', 1e-06)
x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
obj = gurobipy.quicksum((params[i]['r'] * x[i] for i in I))
m.setObjective(obj, GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= params[i]['d'], name='')
    m.addConstr(x[i] <= params[i]['s'], name='')
m.optimize()
