import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
pattern = re.compile('^\\s*(?P<Sub_Category>.+?)\\s+(?P<Revenue>[-+]?\\d*\\.?\\d+)\\s+(?P<Demand>\\d+)\\s+(?P<Initial_Inventory>[-+]?\\d*\\.?\\d+)\\s*$', re.MULTILINE)
matches = list(pattern.finditer(csv_text))
if not matches:
    raise ValueError('No rows matched the expected CSV structure.')
rows = []
for m in matches:
    row = {'Sub Category': m.group('Sub_Category').strip(), 'Revenue': m.group('Revenue').strip(), 'Demand': m.group('Demand').strip(), 'Initial Inventory': m.group('Initial_Inventory').strip()}
    rows.append(row)
df = pd.DataFrame(rows)
df['Revenue'] = pd.to_numeric(df['Revenue'])
df['Demand'] = pd.to_numeric(df['Demand'], downcast='integer')
df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], downcast='integer')
organ_mask = df['Sub Category'].str.contains('Organ', case=False, na=False)
organ_df = df[organ_mask].copy()
if organ_df.empty:
    raise ValueError("No products classified under 'Organ' found in the CSV.")
I = list(organ_df['Sub Category'])
A = dict(zip(organ_df['Sub Category'], organ_df['Revenue']))
d = dict(zip(organ_df['Sub Category'], organ_df['Demand']))
I_i = dict(zip(organ_df['Sub Category'], organ_df['Initial Inventory']))
for i in I:
    if i not in A or i not in d or i not in I_i:
        raise ValueError(f"Missing data for product '{i}'.")
m = gp.Model('Supermart_Organ_Revenue')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I_i[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')