import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Unable to read CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    if not all((col in df.columns for col in required_cols)):
        raise ValueError(f'Missing required columns in CSV. Found columns: {df.columns}')
    faux_mask = df['Product Name'].astype(str).str.casefold().str.contains('faux')
    faux_df = df[faux_mask].copy()
    if faux_df.empty:
        raise ValueError("No 'FAUX' products found in the data.")
    faux_df = faux_df.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = list(faux_df['Product Name'])
    revenue = dict(zip(faux_df['Product Name'], faux_df['Revenue']))
    demand = dict(zip(faux_df['Product Name'], faux_df['Demand']))
    inventory = dict(zip(faux_df['Product Name'], faux_df['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficients for product: {i}')
        if pd.isnull(revenue[i]) or pd.isnull(demand[i]) or pd.isnull(inventory[i]):
            raise ValueError(f'Null coefficient for product: {i}')
        if not (isinstance(revenue[i], (int, float)) and isinstance(demand[i], (int, float)) and isinstance(inventory[i], (int, float))):
            raise ValueError(f'Non-numeric coefficient for product: {i}')
    m = gp.Model('ZARA_FAUX_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()