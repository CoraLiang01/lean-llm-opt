import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
fashion_mask = df['Product Name'].str.contains('Fashion', flags=re.IGNORECASE, regex=True, na=False)
fashion_df = df[fashion_mask].copy()
if fashion_df.empty:
    raise ValueError("No 'Fashion' products found in the dataset.")

def parse_numeric(val, col, idx):
    if not re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', val):
        raise ValueError(f"Unmatched numeric value in column '{col}' for product '{idx}': '{val}'")
    return float(val) if '.' in val else int(val)
F = []
a = {}
d = {}
I = {}
for _, row in fashion_df.iterrows():
    prod = row['Product Name']
    rev = row['Revenue']
    dem = row['Demand']
    inv = row['Initial Inventory']
    a_i = parse_numeric(rev, 'Revenue', prod)
    d_i = parse_numeric(dem, 'Demand', prod)
    I_i = parse_numeric(inv, 'Initial Inventory', prod)
    F.append(prod)
    a[prod] = a_i
    d[prod] = d_i
    I[prod] = I_i
for prod in F:
    if not (prod in a and prod in d and (prod in I)):
        raise ValueError(f"Missing parameter for product '{prod}'.")
m = gp.Model('Fashion_Revenue_Max')
x_vars = m.addVars(F, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((a[i] * x_vars[i] for i in F)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= d[i] for i in F), name='')
m.addConstrs((x_vars[i] <= I[i] for i in F), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')