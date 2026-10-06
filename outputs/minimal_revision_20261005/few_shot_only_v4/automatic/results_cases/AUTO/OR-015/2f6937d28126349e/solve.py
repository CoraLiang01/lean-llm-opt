import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    mask = df['Product Name'].astype(str).str.casefold().str.contains('aalop')
    df_aalop = df[mask].copy()
    if df_aalop.empty:
        raise ValueError("No products classified under 'Aalop' found in the data.")
    df_aalop = df_aalop.dropna(subset=['Product Name', 'Revenue', 'Demand', 'Initial Inventory'])
    items = df_aalop['Product Name'].astype(str).tolist()
    if len(set(items)) != len(items):
        raise ValueError("Duplicate 'Product Name' entries found for 'Aalop' products.")
    try:
        revenue = dict(zip(items, df_aalop['Revenue'].astype(float)))
        demand = dict(zip(items, df_aalop['Demand'].astype(float)))
        inventory = dict(zip(items, df_aalop['Initial Inventory'].astype(float)))
    except Exception as e:
        raise ValueError(f'Error converting parameters to float: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
    m = gp.Model('Aalop_Restaurant_RevOpt')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
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