import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
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
        raise ValueError("No '27in' products found in the data.")
    df_27in = df_27in.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_27in['Product Name'].tolist()
    revenue = dict(zip(df_27in['Product Name'], df_27in['Revenue']))
    demand = dict(zip(df_27in['Product Name'], df_27in['Demand']))
    inventory = dict(zip(df_27in['Product Name'], df_27in['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficients for product: {i}')
        for (dct, name) in [(revenue, 'Revenue'), (demand, 'Demand'), (inventory, 'Initial Inventory')]:
            val = dct[i]
            if not (isinstance(val, (int, float)) and pd.notnull(val)):
                raise ValueError(f'Non-numeric or missing {name} for product: {i}')
    m = gp.Model('Supermarket_27in_Revenue')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()