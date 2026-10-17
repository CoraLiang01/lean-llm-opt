import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV at {csv_path}: {e}')
required_cols = ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in CSV.")
ele_s_mask = df['Product_Reference'].apply(lambda x: bool(re.match('^ELE-S', x)))
df_ele_s = df[ele_s_mask].copy()
if df_ele_s.empty:
    raise ValueError("No products found with Product_Reference starting with 'ELE-S'.")
I = df_ele_s['Product_Reference'].tolist()

def parse_numeric(series, name):
    vals = {}
    for idx, val in zip(df_ele_s['Product_Reference'], series):
        if not re.match('^-?\\d+(\\.\\d+)?$', val.strip()):
            raise ValueError(f"Unmatched or invalid numeric value in column '{name}' for '{idx}': '{val}'")
        vals[idx] = float(val)
    return vals
a_i = parse_numeric(df_ele_s['Revenue'], 'Revenue')
d_i = parse_numeric(df_ele_s['Demand'], 'Demand')
s_i = parse_numeric(df_ele_s['Initial Inventory'], 'Initial Inventory')
for idx in I:
    if idx not in a_i or idx not in d_i or idx not in s_i:
        raise ValueError(f"Missing parameter for product '{idx}'.")
m = gp.Model('ELE_S_Revenue_Max')
x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((a_i[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= d_i[i] for i in I), name='')
m.addConstrs((x_vars[i] <= s_i[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')