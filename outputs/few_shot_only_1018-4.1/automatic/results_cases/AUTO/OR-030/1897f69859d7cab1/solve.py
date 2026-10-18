import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^(Product Name\\s+Revenue\\s+Demand\\s+Initial Inventory)\\n', csv_text, re.MULTILINE)
if not header_match:
    raise RuntimeError('CSV header not found or does not match expected columns.')
header_line = header_match.group(1)
header_cols = [h.strip() for h in re.split('\\s+', header_line)]
if header_cols != ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']:
    raise RuntimeError('CSV header columns do not match expected: %s' % header_cols)
row_pattern = re.compile('^(?P<ProductName>\\S+)\\s+(?P<Revenue>[-+]?\\d*\\.?\\d+)\\s+(?P<Demand>\\d+)\\s+(?P<InitialInventory>\\d+)$', re.MULTILINE)
rows = []
for m in row_pattern.finditer(csv_text):
    row = {'Product Name': m.group('ProductName'), 'Revenue': m.group('Revenue'), 'Demand': m.group('Demand'), 'Initial Inventory': m.group('InitialInventory')}
    rows.append(row)
if not rows:
    raise RuntimeError('No data rows matched the expected pattern in the CSV.')
fdk57_rows = [r for r in rows if r['Product Name'] == 'FDK57']
if not fdk57_rows:
    raise RuntimeError("No rows found for Product Name 'FDK57'.")
I = [r['Product Name'] + f'_{idx}' if len(fdk57_rows) > 1 else r['Product Name'] for idx, r in enumerate(fdk57_rows)]
A = {}
d = {}
I_param = {}
for idx, r in enumerate(fdk57_rows):
    key = r['Product Name'] + f'_{idx}' if len(fdk57_rows) > 1 else r['Product Name']
    try:
        A[key] = float(r['Revenue'])
        d[key] = int(r['Demand'])
        I_param[key] = int(r['Initial Inventory'])
    except Exception as e:
        raise RuntimeError(f'Failed to parse numeric fields for row {r}: {e}')
for key in I:
    if key not in A or key not in d or key not in I_param:
        raise RuntimeError(f'Missing parameter for index {key}')
m = gp.Model('Car_Dealership_FDK57')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I_param[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')