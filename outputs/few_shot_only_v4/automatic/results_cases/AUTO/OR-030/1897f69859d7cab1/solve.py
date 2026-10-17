import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV: {e}')
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
df_fdk57 = df[df['Product Name'] == 'FDK57'].copy()
if df_fdk57.empty:
    raise ValueError("No rows found with Product Name == 'FDK57'.")
I = list(df_fdk57.index)

def parse_numeric_column(series, colname):
    values = {}
    for idx, val in series.items():
        m = re.fullmatch('\\s*(-?\\d+(?:\\.\\d+)?)\\s*', val)
        if not m:
            raise ValueError(f"Unmatched or invalid numeric value in column '{colname}' at row {idx}: '{val}'")
        values[idx] = float(m.group(1))
    return values
A = parse_numeric_column(df_fdk57['Revenue'], 'Revenue')
d = parse_numeric_column(df_fdk57['Demand'], 'Demand')
I_param = parse_numeric_column(df_fdk57['Initial Inventory'], 'Initial Inventory')
if not set(A) == set(d) == set(I_param) == set(I):
    raise ValueError('Index sets for parameters do not match.')
m = gp.Model('FDK57_Car_Dealership_RevMax')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name='')
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