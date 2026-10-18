import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV: {e}')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
pattern = re.compile('27in', re.IGNORECASE)
mask = df['Product Name'].apply(lambda x: bool(pattern.search(x)))
if not mask.any():
    raise ValueError("No products found with '27in' in Product Name.")
I = df.loc[mask, 'Product Name'].tolist()
if not I:
    raise ValueError('No eligible products in index set I.')

def parse_numeric(series, name):
    vals = {}
    for idx, val in zip(df.loc[mask, 'Product Name'], series.loc[mask]):
        try:
            vals[idx] = float(val)
        except Exception:
            raise ValueError(f"Non-numeric or missing value in column '{name}' for product '{idx}': '{val}'")
    return vals
A = parse_numeric(df['Revenue'], 'Revenue')
d = parse_numeric(df['Demand'], 'Demand')
I_i = parse_numeric(df['Initial Inventory'], 'Initial Inventory')
for prod in I:
    for param, param_dict in [('Revenue', A), ('Demand', d), ('Initial Inventory', I_i)]:
        if prod not in param_dict:
            raise ValueError(f"Missing {param} for product '{prod}'.")
m = gp.Model('NRM_27in_Revenue')
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I_i[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')