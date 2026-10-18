import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV: {e}')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
baby_mask = df['Product Name'].apply(lambda x: bool(re.search('\\bBaby\\b', x)))
if not baby_mask.any():
    raise ValueError("No products classified under 'Baby' found in the CSV.")
I = df.loc[baby_mask, 'Product Name'].tolist()
if not I:
    raise ValueError("No eligible 'Baby' products found after filtering.")

def parse_float(val, field, prod):
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Could not parse {field} for product '{prod}': '{val}'")
revenue = {}
demand = {}
inventory = {}
for idx, row in df.loc[baby_mask].iterrows():
    prod = row['Product Name']
    revenue[prod] = parse_float(row['Revenue'], 'Revenue', prod)
    demand[prod] = int(parse_float(row['Demand'], 'Demand', prod))
    inventory[prod] = int(parse_float(row['Initial Inventory'], 'Initial Inventory', prod))
for prod in I:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f"Missing parameter for product '{prod}'.")
m = gp.Model('Baby_Product_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    upper = min(inventory[i], demand[i])
    m.addConstr(x_vars[i] <= upper, name=f'ub_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'lb_{i}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')