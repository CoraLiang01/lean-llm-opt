import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV at {csv_path}: {e}')
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in CSV.")
pattern = re.compile('27in', re.IGNORECASE)
mask = df['Product Name'].apply(lambda x: bool(pattern.search(x)))
if not mask.any():
    raise ValueError("No products found with '27in' in the Product Name column.")
df_27in = df[mask].copy()
items = df_27in['Product Name'].tolist()

def parse_numeric(series, colname):
    vals = {}
    for idx, (k, v) in enumerate(zip(df_27in['Product Name'], series)):
        v_stripped = str(v).strip()
        if not re.fullmatch('-?\\d+(\\.\\d+)?', v_stripped):
            raise ValueError(f"Unmatched or invalid numeric value in column '{colname}' for product '{k}': '{v}'")
        try:
            if '.' in v_stripped:
                vals[k] = float(v_stripped)
            else:
                vals[k] = int(v_stripped)
        except Exception as e:
            raise ValueError(f"Failed to parse numeric value in column '{colname}' for product '{k}': '{v}' ({e})")
    return vals
revenue = parse_numeric(df_27in['Revenue'], 'Revenue')
demand = parse_numeric(df_27in['Demand'], 'Demand')
inventory = parse_numeric(df_27in['Initial Inventory'], 'Initial Inventory')
for k in items:
    if k not in revenue or k not in demand or k not in inventory:
        raise ValueError(f"Missing data for product '{k}' in one or more required columns.")
m = gp.Model('NRM_27in')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')