import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    cap_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
    prod_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
    for enc in csv_encodings:
        try:
            cap_df = pd.read_csv(cap_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {cap_path} with tried encodings.')
    for enc in csv_encodings:
        try:
            prod_df = pd.read_csv(prod_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {prod_path} with tried encodings.')
    if not {'DisplayID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    I = cap_df['DisplayID'].astype(str).unique().tolist()
    J = prod_df['ProductName'].astype(str).unique().tolist()
    c_i = cap_df.groupby('DisplayID')['Capacity'].sum().to_dict()
    v_j = prod_df.groupby('ProductName')['Value'].sum().to_dict()
    w_j = prod_df.groupby('ProductName')['Weight'].sum().to_dict()
    for i in I:
        if i not in c_i:
            raise ValueError(f'Missing capacity for DisplayID {i}')
    for j in J:
        if j not in v_j or j not in w_j:
            raise ValueError(f'Missing value or weight for ProductName {j}')
    m = gp.Model('BoatDisplayAssignment')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in I for j in J]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * x[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')