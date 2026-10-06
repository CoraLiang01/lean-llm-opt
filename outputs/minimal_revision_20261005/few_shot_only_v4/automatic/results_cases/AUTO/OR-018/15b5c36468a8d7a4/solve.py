import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    if 'Category' in df.columns:
        baby_mask = df['Category'].str.casefold() == 'baby'
    else:
        raise ValueError("Source data must include a 'Category' column to identify 'Baby' products.")
    baby_df = df[baby_mask].copy()
    if baby_df.empty:
        raise ValueError("No products classified as 'Baby' found in the source data.")
    group_cols = ['Product Name']
    agg_dict = {'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'}
    baby_df = baby_df.groupby(group_cols, as_index=False).agg(agg_dict)
    items = list(baby_df['Product Name'])
    revenue = dict(zip(baby_df['Product Name'], baby_df['Revenue']))
    demand = dict(zip(baby_df['Product Name'], baby_df['Demand']))
    inventory = dict(zip(baby_df['Product Name'], baby_df['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficients for product: {i}')
        for (d, name) in zip([revenue[i], demand[i], inventory[i]], ['Revenue', 'Demand', 'Initial Inventory']):
            if pd.isnull(d):
                raise ValueError(f'Missing {name} for product: {i}')
            if not isinstance(d, (int, float)):
                raise ValueError(f'Non-numeric {name} for product: {i}')
    m = gp.Model('Baby_Product_Revenue_Max')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
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