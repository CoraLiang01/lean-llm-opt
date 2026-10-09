import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == encodings[-1]:
                raise
            continue
    mask = df['Sub Category'].str.casefold().str.contains('organ')
    organ_df = df[mask].copy()
    required_cols = ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in organ_df.columns:
            raise ValueError(f'Missing required column: {col}')
    I = organ_df.index.tolist()
    try:
        A_i = organ_df['Revenue'].astype(float).to_dict()
        d_i = organ_df['Demand'].astype(float).to_dict()
        I_i = organ_df['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting parameters to numeric: {e}')
    for i in I:
        if i not in A_i or i not in d_i or i not in I_i:
            raise ValueError(f'Missing parameter for index {i}')
    m = gp.Model('Supermart_Organ_Revenue')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((A_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= I_i[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= d_i[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()