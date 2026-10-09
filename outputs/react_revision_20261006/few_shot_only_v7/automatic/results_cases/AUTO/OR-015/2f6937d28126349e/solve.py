import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df['Product Name'] = df['Product Name'].str.strip()
    group_cols = ['Product Name']
    agg_dict = {'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'}
    df_grouped = df.groupby('Product Name', as_index=False).agg(agg_dict)
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        try:
            df_grouped[col] = pd.to_numeric(df_grouped[col], errors='raise')
        except Exception as e:
            raise ValueError(f'Non-numeric value in column {col}: {e}')
    items = df_grouped['Product Name'].tolist()
    revenue = dict(zip(df_grouped['Product Name'], df_grouped['Revenue']))
    demand = dict(zip(df_grouped['Product Name'], df_grouped['Demand']))
    inventory = dict(zip(df_grouped['Product Name'], df_grouped['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficient for product: {i}')
    m = gp.Model('Restaurant_Aalop_RevMax')
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()