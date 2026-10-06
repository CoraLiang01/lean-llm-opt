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
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    if 'id_number' not in df.columns:
        raise ValueError('Missing required column: id_number')
    mask = df['id_number'].astype(str).str.casefold() == 'id999'
    df_id999 = df[mask].copy()
    if df_id999.empty:
        raise ValueError("No rows found with id_number == 'id999'.")
    required_cols = ['Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_id999.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(df_id999.index)
    try:
        a = df_id999['Revenue'].to_dict()
        d = df_id999['Demand'].to_dict()
        s = df_id999['Initial Inventory'].to_dict()
    except Exception as e:
        raise ValueError(f'Error extracting parameters: {e}')
    for i in I:
        if pd.isnull(a[i]) or pd.isnull(d[i]) or pd.isnull(s[i]):
            raise ValueError(f'Missing parameter for product index {i}')
    for i in I:
        if not (isinstance(a[i], (int, float)) and isinstance(d[i], (int, float)) and isinstance(s[i], (int, float))):
            raise ValueError(f'Non-numeric parameter for product index {i}')
        if d[i] < 0 or s[i] < 0 or a[i] < 0:
            raise ValueError(f'Negative parameter for product index {i}')
    m = gp.Model('NRM_id999')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((a[i] * x[i] for i in I)), GRB.MAXIMIZE)
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