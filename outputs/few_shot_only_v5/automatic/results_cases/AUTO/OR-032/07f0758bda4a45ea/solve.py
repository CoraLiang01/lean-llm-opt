import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^\\s*Product_Name\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*$', csv_text, re.MULTILINE)
if not header_match:
    raise RuntimeError('CSV header not found or does not match required columns.')
row_pattern = re.compile('^\\s*(?P<Product_Name>[^\\n]+?)\\s+(?P<Revenue>[0-9.]+)\\s+(?P<Demand>[0-9]+)\\s+(?P<Initial_Inventory>[0-9.]+)\\s*$', re.MULTILINE)
rows = [m.groupdict() for m in row_pattern.finditer(csv_text)]
if not rows:
    raise RuntimeError('No data rows matched the expected CSV format.')
df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError('No rows parsed into DataFrame.')
df['Revenue'] = df['Revenue'].astype(float)
df['Demand'] = df['Demand'].astype(int)
df['Initial_Inventory'] = df['Initial_Inventory'].astype(float)
books_mask = df['Product_Name'].str.contains('Books', regex=False)
books_df = df[books_mask].copy()
if books_df.empty:
    raise RuntimeError("No products found with 'Books' in Product_Name.")
I = list(books_df['Product_Name'])
A = dict(zip(books_df['Product_Name'], books_df['Revenue']))
d = dict(zip(books_df['Product_Name'], books_df['Demand']))
I_param = dict(zip(books_df['Product_Name'], books_df['Initial_Inventory']))
for i in I:
    if i not in A or i not in d or i not in I_param:
        raise RuntimeError(f'Missing parameter for product {i}.')
m = gp.Model('Books_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.addConstrs((x_vars[i] <= I_param[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')