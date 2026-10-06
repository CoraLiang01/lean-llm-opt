import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    mask = df['Sub Category'].astype(str).str.casefold() == 'organ'
    organ_df = df[mask].copy()
    required_cols = ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in organ_df.columns:
            raise ValueError(f'Missing required column: {col}')
    organ_df = organ_df.dropna(subset=['Revenue', 'Demand', 'Initial Inventory'])
    items = organ_df.index.tolist()
    try:
        revenue = organ_df['Revenue'].to_dict()
        demand = organ_df['Demand'].to_dict()
        inventory = organ_df['Initial Inventory'].to_dict()
    except Exception as e:
        raise ValueError(f'Error extracting parameters: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficients for item {i}')
    m = gp.Model('Supermart_Organ_Revenue')
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