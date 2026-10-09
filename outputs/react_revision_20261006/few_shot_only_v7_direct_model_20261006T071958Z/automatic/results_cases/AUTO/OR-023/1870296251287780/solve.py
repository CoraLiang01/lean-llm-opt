import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == encodings[-1]:
                raise
            continue
    mask = df['Product_Reference'].str.casefold().str.startswith('ele-s')
    ele_s_df = df[mask].copy()
    required_cols = ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in ele_s_df.columns:
            raise ValueError(f'Missing required column: {col}')
    ele_s_df = ele_s_df.reset_index(drop=False)
    ele_s_df['key'] = list(zip(ele_s_df['index'], ele_s_df['Product_Reference']))
    keys = ele_s_df['key'].tolist()
    product_reference = dict(zip(keys, ele_s_df['Product_Reference']))

    def to_int_series(series, colname):
        try:
            return series.astype(str).str.replace(',', '').astype(float).astype(int)
        except Exception:
            raise ValueError(f"Non-numeric or missing value in column '{colname}' for some ELE-S products.")
    revenue = dict(zip(keys, to_int_series(ele_s_df['Revenue'], 'Revenue')))
    demand = dict(zip(keys, to_int_series(ele_s_df['Demand'], 'Demand')))
    inventory = dict(zip(keys, to_int_series(ele_s_df['Initial Inventory'], 'Initial Inventory')))
    for k in keys:
        if k not in revenue or k not in demand or k not in inventory:
            raise ValueError(f'Missing parameter for key {k}')
    m = gp.Model('ELE_S_Revenue_Max')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[k] * quantity_vars[k] for k in keys)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[k] <= inventory[k] for k in keys), name='')
    m.addConstrs((quantity_vars[k] <= demand[k] for k in keys), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for k in keys:
            print(f'x[{product_reference[k]}|row{str(k[0])}]: {quantity_vars[k].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()