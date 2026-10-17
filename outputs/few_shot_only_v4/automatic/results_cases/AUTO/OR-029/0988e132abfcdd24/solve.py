import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
faux_rows = []
for idx, row in df.iterrows():
    matched = False
    for val in row.values:
        if re.fullmatch('\\s*FAUX\\s*', str(val), re.IGNORECASE):
            matched = True
            break
    if matched:
        faux_rows.append(idx)
if not faux_rows:
    raise ValueError("No products classified under 'FAUX' found in the dataset.")
I = faux_rows
a_i = {}
d_i = {}
s_i = {}
for i in I:
    try:
        revenue_str = df.at[i, 'Revenue']
        demand_str = df.at[i, 'Demand']
        inventory_str = df.at[i, 'Initial Inventory']
    except KeyError as e:
        raise ValueError(f'Missing required data for row {i}: {e}')
    if not re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', revenue_str):
        raise ValueError(f"Revenue value '{revenue_str}' in row {i} is not a valid number.")
    if not re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', demand_str):
        raise ValueError(f"Demand value '{demand_str}' in row {i} is not a valid number.")
    if not re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', inventory_str):
        raise ValueError(f"Initial Inventory value '{inventory_str}' in row {i} is not a valid number.")
    a_i[i] = float(revenue_str)
    d_i[i] = int(float(demand_str))
    s_i[i] = int(float(inventory_str))
if set(a_i.keys()) != set(I) or set(d_i.keys()) != set(I) or set(s_i.keys()) != set(I):
    raise ValueError('Parameter coverage mismatch for index set I.')
m = gp.Model('ZARA_FAUX_Revenue_Max')
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((a_i[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= s_i[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d_i[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')