import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV at {csv_path}: {e}')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in CSV.")
baby_mask = df['Product Name'].str.contains('\\bBaby\\b', flags=re.IGNORECASE, regex=True)
baby_df = df[baby_mask].copy()
if baby_df.empty:
    raise ValueError("No products classified as 'Baby' found in the CSV.")

def parse_numeric(series, colname):
    vals = []
    for idx, val in series.items():
        try:
            v = float(val)
        except Exception:
            raise ValueError(f"Non-numeric value '{val}' in column '{colname}' for row '{idx}'.")
        vals.append(v)
    return vals
product_names = list(baby_df['Product Name'])
revenue_vals = parse_numeric(baby_df['Revenue'], 'Revenue')
demand_vals = parse_numeric(baby_df['Demand'], 'Demand')
inventory_vals = parse_numeric(baby_df['Initial Inventory'], 'Initial Inventory')
revenue = dict(zip(product_names, revenue_vals))
demand = dict(zip(product_names, demand_vals))
inventory = dict(zip(product_names, inventory_vals))
if not set(revenue) == set(demand) == set(inventory) == set(product_names):
    raise ValueError('Mismatch in product indices among parameters.')
m = gp.Model('Baby_Product_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in product_names), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in product_names), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')