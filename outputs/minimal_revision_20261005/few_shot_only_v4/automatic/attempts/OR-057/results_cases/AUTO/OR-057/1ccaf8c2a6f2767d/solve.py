import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv'
    try:
        cap = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            cap = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                cap = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                cap = pd.read_csv(capacity_path, encoding='latin-1')
    if not {'PlatformID', 'Capacity'}.issubset(cap.columns):
        raise ValueError('capacity.csv must contain columns: PlatformID, Capacity')
    platforms = cap['PlatformID'].astype(str).tolist()
    c = cap.set_index('PlatformID')['Capacity'].to_dict()
    for pid in platforms:
        if pid not in c or pd.isnull(c[pid]):
            raise ValueError(f'Missing capacity for platform {pid}')
        try:
            c[pid] = float(c[pid])
        except Exception:
            raise ValueError(f'Non-numeric capacity for platform {pid}: {c[pid]}')
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv'
    try:
        prod = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            prod = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                prod = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                prod = pd.read_csv(products_path, encoding='latin-1')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod.columns):
        raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
    games = prod['ProductName'].astype(str).tolist()
    v = prod.set_index('ProductName')['Value'].to_dict()
    w = prod.set_index('ProductName')['Weight'].to_dict()
    for g in games:
        if g not in v or pd.isnull(v[g]):
            raise ValueError(f'Missing value for game {g}')
        if g not in w or pd.isnull(w[g]):
            raise ValueError(f'Missing weight for game {g}')
        try:
            v[g] = float(v[g])
        except Exception:
            raise ValueError(f'Non-numeric value for game {g}: {v[g]}')
        try:
            w[g] = float(w[g])
        except Exception:
            raise ValueError(f'Non-numeric weight for game {g}: {w[g]}')
    m = gp.Model('VideoGame_Platform_Allocation')
    m.setParam('MIPGap', 0.0001)
    keys = [(i, j) for i in platforms for j in games]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * x[i, j] for i in platforms for j in games)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * x[i, j] for j in games)) <= c[i] for i in platforms), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')