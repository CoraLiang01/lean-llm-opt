import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^\\s*id_number\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*$', csv_text, re.MULTILINE)
if not header_match:
    raise ValueError('Header row not found or does not match expected format.')
row_pattern = re.compile('^\\s*(id\\d+)\\s+([0-9.]+)\\s+([0-9]+)\\s+([0-9]+)\\s*$', re.MULTILINE)
rows = row_pattern.findall(csv_text)
if not rows:
    raise ValueError('No data rows matched the expected pattern.')
df = pd.DataFrame(rows, columns=['id_number', 'Revenue', 'Demand', 'Initial Inventory'])
df['Revenue'] = df['Revenue'].astype(float)
df['Demand'] = df['Demand'].astype(int)
df['Initial Inventory'] = df['Initial Inventory'].astype(int)
df_id999 = df[df['id_number'] == 'id999']
if df_id999.empty:
    raise ValueError("No products with id_number == 'id999' found in the dataset.")
I = df_id999['id_number'].tolist()
if len(I) != 1:
    raise ValueError("Expected exactly one product with id_number == 'id999', found %d." % len(I))
a = {row['id_number']: row['Revenue'] for _, row in df_id999.iterrows()}
d = {row['id_number']: row['Demand'] for _, row in df_id999.iterrows()}
s = {row['id_number']: row['Initial Inventory'] for _, row in df_id999.iterrows()}
for i in I:
    if i not in a or i not in d or i not in s:
        raise ValueError(f'Missing parameter for product {i}.')
m = gp.Model('Supermarket_OnlineRetailSales_id999')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((a[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= s[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')