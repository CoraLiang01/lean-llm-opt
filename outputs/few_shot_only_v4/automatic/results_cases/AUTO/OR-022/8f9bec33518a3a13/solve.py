import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV at {csv_path}: {e}')
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
pattern = re.compile('27in')
mask = df['Product Name'].apply(lambda x: bool(pattern.search(x)))
df_27in = df[mask].copy()
if df_27in.empty:
    raise ValueError("No products found with '27in' in the Product Name column.")
items = df_27in['Product Name'].tolist()
revenue = {}
demand = {}
inventory = {}
for idx, row in df_27in.iterrows():
    name = row['Product Name']
    try:
        revenue_val = float(row['Revenue'])
    except Exception:
        raise ValueError(f"Unmatched or invalid Revenue for product '{name}'.")
    try:
        demand_val = int(float(row['Demand']))
    except Exception:
        raise ValueError(f"Unmatched or invalid Demand for product '{name}'.")
    try:
        inventory_val = int(float(row['Initial Inventory']))
    except Exception:
        raise ValueError(f"Unmatched or invalid Initial Inventory for product '{name}'.")
    revenue[name] = revenue_val
    demand[name] = demand_val
    inventory[name] = inventory_val
if set(items) != set(revenue.keys()) or set(items) != set(demand.keys()) or set(items) != set(inventory.keys()):
    raise ValueError('Mismatch in index sets between items and parameter dictionaries.')
m = gp.Model('NRM_27in_Fulfillment')
x_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')