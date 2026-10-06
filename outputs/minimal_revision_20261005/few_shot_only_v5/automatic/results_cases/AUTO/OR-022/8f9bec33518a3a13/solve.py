import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    mask_27in = df['Product Name'].astype(str).str.casefold().str.contains('27in')
    df_27in = df[mask_27in].copy()
    if df_27in.empty:
        raise ValueError("No '27in' products found in the source data.")
    df_agg = df_27in.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_agg['Product Name'].tolist()
    revenue = dict(zip(df_agg['Product Name'], df_agg['Revenue']))
    demand = dict(zip(df_agg['Product Name'], df_agg['Demand']))
    inventory = dict(zip(df_agg['Product Name'], df_agg['Initial Inventory']))
    for i in items:
        if pd.isnull(revenue[i]) or pd.isnull(demand[i]) or pd.isnull(inventory[i]):
            raise ValueError(f'Missing data for product: {i}')
        if not (isinstance(revenue[i], (int, float)) and isinstance(demand[i], (int, float)) and isinstance(inventory[i], (int, float))):
            raise ValueError(f'Non-numeric data for product: {i}')
    var_ub = {i: min(int(demand[i]), int(inventory[i])) for i in items}
    m = gp.Model('NRM_27in')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, ub=var_ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()