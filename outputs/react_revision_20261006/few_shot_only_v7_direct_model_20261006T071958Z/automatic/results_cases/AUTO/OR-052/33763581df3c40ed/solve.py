import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    cap_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
    prod_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
    for enc in csv_encodings:
        try:
            cap_df = pd.read_csv(cap_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cap_path} with tried encodings.')
    for enc in csv_encodings:
        try:
            prod_df = pd.read_csv(prod_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {prod_path} with tried encodings.')
    if 'BookshelfID' not in cap_df.columns or 'Capacity' not in cap_df.columns:
        raise ValueError("capacity.csv must have columns 'BookshelfID' and 'Capacity'")
    if 'ProductName' not in prod_df.columns or 'Value' not in prod_df.columns or 'Weight' not in prod_df.columns:
        raise ValueError("products.csv must have columns 'ProductName', 'Value', 'Weight'")
    I = cap_df['BookshelfID'].tolist()
    J = prod_df['ProductName'].tolist()
    try:
        c_i = cap_df.set_index('BookshelfID')['Capacity'].astype(float).to_dict()
    except Exception:
        raise ValueError("Non-numeric value in 'Capacity' column of capacity.csv")
    try:
        v_j = prod_df.set_index('ProductName')['Value'].astype(float).to_dict()
        w_j = prod_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception:
        raise ValueError("Non-numeric value in 'Value' or 'Weight' column of products.csv")
    if len(I) != len(set(I)):
        raise ValueError('Duplicate BookshelfID in capacity.csv')
    if len(J) != len(set(J)):
        raise ValueError('Duplicate ProductName in products.csv')
    for i in I:
        if i not in c_i:
            raise ValueError(f'Missing capacity for BookshelfID {i}')
    for j in J:
        if j not in v_j or j not in w_j:
            raise ValueError(f'Missing value or weight for ProductName {j}')
    m = gp.Model('Bookshelf_Allocation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')