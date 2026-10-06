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
    mask = df['Product Name'].astype(str).str.casefold().str.contains('27in')
    df_27in = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_27in.columns:
            raise ValueError(f'Missing required column: {col}')
    I = df_27in['Product Name'].astype(str).tolist()
    grouped = df_27in.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    A = dict(zip(grouped['Product Name'], grouped['Revenue']))
    d = dict(zip(grouped['Product Name'], grouped['Demand']))
    I_i = dict(zip(grouped['Product Name'], grouped['Initial Inventory']))
    for prod in grouped['Product Name']:
        if prod not in A or prod not in d or prod not in I_i:
            raise ValueError(f'Missing parameter for product: {prod}')
    m = gp.Model('27in_Product_Revenue_Max')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(grouped['Product Name'], lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((A[i] * x[i] for i in grouped['Product Name'])), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= d[i] for i in grouped['Product Name']), name='')
    m.addConstrs((x[i] <= I_i[i] for i in grouped['Product Name']), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()