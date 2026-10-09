import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'file_0_view_0': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', 'file_1_view_0': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_fallback(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    df_platforms = read_csv_with_fallback(csv_paths['file_0_view_0'])
    df_products = read_csv_with_fallback(csv_paths['file_1_view_0'])
    platforms = df_platforms['PlatformId'].unique().tolist()
    genres = df_products['ProductName'].unique().tolist()
    c_i = {}
    for (_, row) in df_platforms.iterrows():
        pid = row['PlatformId']
        if pid not in c_i:
            try:
                c_i[pid] = float(row['Capacity'])
            except Exception:
                raise ValueError(f"Invalid Capacity for PlatformId {pid}: {row['Capacity']}")
    v_j = {}
    for (_, row) in df_products.iterrows():
        pname = row['ProductName']
        if pname not in v_j:
            try:
                v_j[pname] = float(row['Value'])
            except Exception:
                raise ValueError(f"Invalid Value for ProductName {pname}: {row['Value']}")
    w_j = {}
    for (_, row) in df_products.iterrows():
        pname = row['ProductName']
        if pname not in w_j:
            try:
                w_j[pname] = float(row['Weight'])
            except Exception:
                raise ValueError(f"Invalid Weight for ProductName {pname}: {row['Weight']}")
    missing_platforms = set(platforms) - set(c_i)
    missing_genres = set(genres) - set(v_j) - set(w_j)
    if missing_platforms:
        raise ValueError(f'Missing capacity for platforms: {missing_platforms}')
    if set(genres) - set(v_j):
        raise ValueError(f'Missing value for genres: {set(genres) - set(v_j)}')
    if set(genres) - set(w_j):
        raise ValueError(f'Missing weight for genres: {set(genres) - set(w_j)}')
    keys = [(i, j) for i in platforms for j in genres]
    m = gp.Model('VideoGameStore_Listing')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
    for i in platforms:
        m.addConstr(gp.quicksum((w_j[j] * quantity_vars[i, j] for j in genres)) <= c_i[i])
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')