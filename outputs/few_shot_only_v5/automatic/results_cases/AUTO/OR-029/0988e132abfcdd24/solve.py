import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
if not all((col in df.columns for col in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'])):
    raise ValueError('CSV missing required columns.')
faux_mask = df['Product Name'].str.contains('\\bFAUX\\b', case=False, regex=True) | df['Product Name'].str.contains('FAUX', case=False, regex=False)
df_faux = df[faux_mask].copy()
if df_faux.empty:
    raise ValueError("No products classified as 'FAUX' found in the dataset.")

def parse_float(val, name, idx):
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Non-numeric {name} for product '{idx}': '{val}'")

def parse_int(val, name, idx):
    try:
        return int(float(val))
    except Exception:
        raise ValueError(f"Non-integer {name} for product '{idx}': '{val}'")
products = df_faux['Product Name'].tolist()
revenue = {}
demand = {}
inventory = {}
for _, row in df_faux.iterrows():
    idx = row['Product Name']
    revenue[idx] = parse_float(row['Revenue'], 'Revenue', idx)
    demand[idx] = parse_int(row['Demand'], 'Demand', idx)
    inventory[idx] = parse_int(row['Initial Inventory'], 'Initial Inventory', idx)
if not set(products) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
    raise ValueError('Mismatch in product indices and parameter keys.')
m = gp.Model('ZARA_FAUX_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')