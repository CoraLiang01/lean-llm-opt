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
        raise RuntimeError('Could not read CSV with any of the specified encodings.')
    mask = df['Product Name'].astype(str).str.casefold().str.contains('27in')
    df_27in = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_27in.columns:
            raise ValueError(f'Missing required column: {col}')
    group_cols = ['Product Name']
    agg_dict = {'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'}
    df_agg = df_27in.groupby(group_cols, as_index=False).agg(agg_dict)
    items = df_agg['Product Name'].tolist()
    if not items:
        raise ValueError("No products with '27in' found in 'Product Name'.")
    try:
        revenue = {row['Product Name']: float(row['Revenue']) for (_, row) in df_agg.iterrows()}
        demand = {row['Product Name']: int(row['Demand']) for (_, row) in df_agg.iterrows()}
        inventory = {row['Product Name']: int(row['Initial Inventory']) for (_, row) in df_agg.iterrows()}
    except Exception as e:
        raise ValueError(f'Error processing parameter values: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
        if not (isinstance(revenue[i], (int, float)) and isinstance(demand[i], int) and isinstance(inventory[i], int)):
            raise ValueError(f'Invalid parameter type for product: {i}')
    m = gp.Model('NRM_27in')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()