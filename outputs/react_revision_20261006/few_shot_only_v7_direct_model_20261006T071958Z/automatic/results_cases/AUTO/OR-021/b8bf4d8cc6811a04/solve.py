import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df['Product Name'] = df['Product Name'].str.strip()
    items = df['Product Name'].unique().tolist()
    grouped = df.groupby('Product Name', sort=False, as_index=False).agg({'Revenue': lambda x: sum((float(v) for v in x if v.strip() != '')), 'Demand': lambda x: sum((float(v) for v in x if v.strip() != '')), 'Initial Inventory': lambda x: sum((float(v) for v in x if v.strip() != ''))})
    try:
        revenue = {row['Product Name']: float(row['Revenue']) for (_, row) in grouped.iterrows()}
        demand = {row['Product Name']: int(float(row['Demand'])) for (_, row) in grouped.iterrows()}
        inventory = {row['Product Name']: int(float(row['Initial Inventory'])) for (_, row) in grouped.iterrows()}
    except Exception as e:
        raise ValueError(f'Error converting parameters: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
    m = gp.Model('NRM_Summer_Clothes')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')