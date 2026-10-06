import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Failed to read CSV with supported encodings.')
    if 'id_number' not in df.columns:
        raise ValueError('Missing required column: id_number')
    id_mask = df['id_number'].astype(str).str.casefold() == 'id999'
    df_id999 = df.loc[id_mask].copy()
    required_cols = ['id_number', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_id999.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(df_id999.index)
    a = {}
    s = {}
    d = {}
    for (idx, row) in df_id999.iterrows():
        try:
            a[idx] = float(row['Revenue'])
            s[idx] = int(row['Initial Inventory'])
            d[idx] = int(row['Demand'])
        except Exception as e:
            raise ValueError(f'Invalid data in row {idx}: {e}')
    if not set(I) == set(a.keys()) == set(s.keys()) == set(d.keys()):
        raise ValueError('Mismatch in index sets for coefficients.')
    m = gp.Model('Supermarket_id999_Allocation')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((a[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'x[{i}]: {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()