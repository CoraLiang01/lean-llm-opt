import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cap_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
    prod_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            cap_df = pd.read_csv(cap_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Could not read capacity.csv with supported encodings.')
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            prod_df = pd.read_csv(prod_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Could not read products.csv with supported encodings.')
    if 'Capacity' not in cap_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in prod_df.columns:
            raise ValueError(f"Missing '{col}' column in products.csv")
    if len(cap_df) != 1:
        raise ValueError('capacity.csv must have exactly one row for total capacity.')
    C = cap_df.iloc[0]['Capacity']
    if pd.isnull(C):
        raise ValueError('Capacity value is missing in capacity.csv.')
    prod_df = prod_df.copy()
    prod_df['ProductName'] = prod_df['ProductName'].astype(str)
    prod_df['Value'] = pd.to_numeric(prod_df['Value'], errors='raise')
    prod_df['Weight'] = pd.to_numeric(prod_df['Weight'], errors='raise')
    prod_grouped = prod_df.groupby('ProductName', as_index=False).agg({'Value': 'sum', 'Weight': 'sum'})
    items = list(prod_grouped['ProductName'])
    profit = dict(zip(prod_grouped['ProductName'], prod_grouped['Value']))
    weight = dict(zip(prod_grouped['ProductName'], prod_grouped['Weight']))
    if any((pd.isnull(profit[i]) or pd.isnull(weight[i]) for i in items)):
        raise ValueError('Missing Value or Weight for some products.')
    m = gp.Model('BakeryOrder')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((profit[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in items)) <= C, name='storage')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()