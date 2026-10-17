import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['id_number', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
id999_rows = df[df['id_number'] == 'id999']
if id999_rows.empty:
    raise ValueError("No rows found with id_number == 'id999'.")

def parse_numeric(val, colname, idx):
    if not re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', val):
        raise ValueError(f"Unmatched or invalid numeric value in column '{colname}' for index '{idx}': '{val}'")
    return float(val.strip())
I = list(id999_rows.index)
A = {}
d = {}
I_param = {}
for idx, row in id999_rows.iterrows():
    try:
        A[idx] = parse_numeric(row['Revenue'], 'Revenue', idx)
        d[idx] = int(parse_numeric(row['Demand'], 'Demand', idx))
        I_param[idx] = int(parse_numeric(row['Initial Inventory'], 'Initial Inventory', idx))
    except Exception as e:
        raise ValueError(f'Error parsing row {idx}: {e}')
for idx in I:
    if idx not in A or idx not in d or idx not in I_param:
        raise ValueError(f'Missing parameter for index {idx}')
m = gp.Model('NRM_id999')
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I_param[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')