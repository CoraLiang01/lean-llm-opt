import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    ele_s_mask = None
    for col in df.columns:
        if df[col].dtype == object and df[col].str.casefold().str.contains('ele-s').any():
            ele_s_mask = df[col].str.casefold() == 'ele-s'
            break
    if ele_s_mask is None:
        raise ValueError("No column found with 'ELE-S' classification. Please check the source data.")
    df_ele_s = df[ele_s_mask].copy()
    required_cols = ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_ele_s.columns:
            raise ValueError(f"Missing required column '{col}' in filtered data.")
    df_ele_s = df_ele_s.dropna(subset=required_cols)
    I = df_ele_s['Product_Reference'].astype(str).tolist()
    r = dict(zip(I, df_ele_s['Revenue']))
    d = dict(zip(I, df_ele_s['Demand']))
    s = dict(zip(I, df_ele_s['Initial Inventory']))
    for i in I:
        if not (i in r and i in d and (i in s)):
            raise ValueError(f'Missing data for product {i}.')
        for (param, name) in zip([r[i], d[i], s[i]], ['Revenue', 'Demand', 'Initial Inventory']):
            if not pd.api.types.is_number(param):
                raise ValueError(f'Non-numeric {name} for product {i}.')
    m = gp.Model('ELE_S_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()