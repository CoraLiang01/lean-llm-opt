import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for enc in encodings:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    if not {'BookshelfID', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    B = capacity_df['BookshelfID'].astype(str).unique().tolist()
    P = products_df['ProductName'].astype(str).unique().tolist()
    C_b = {}
    for (_, row) in capacity_df.iterrows():
        b = str(row['BookshelfID'])
        if b in C_b:
            C_b[b] += row['Capacity']
        else:
            C_b[b] = row['Capacity']
    v_p = {}
    w_p = {}
    for (_, row) in products_df.iterrows():
        p = str(row['ProductName'])
        v = row['Value']
        w = row['Weight']
        if p in v_p:
            v_p[p] += v
            w_p[p] += w
        else:
            v_p[p] = v
            w_p[p] = w
    if set(B) != set(C_b.keys()):
        raise ValueError('Mismatch in bookshelf IDs between index set and capacity dictionary.')
    if set(P) != set(v_p.keys()) or set(P) != set(w_p.keys()):
        raise ValueError('Mismatch in product names between index set and value/weight dictionaries.')
    m = gp.Model('Bookshelf_Allocation')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(b, p) for b in B for p in P]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[b, p] for b in B for p in P)), GRB.MAXIMIZE)
    for b in B:
        m.addConstr(gp.quicksum((w_p[p] * x[b, p] for p in P)) <= C_b[b], name=f'cap_{b}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()