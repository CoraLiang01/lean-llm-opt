import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    decode_attempts = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in decode_attempts:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    I = cap_df['DisplayID'].tolist()
    J = prod_df['ProductName'].tolist()
    try:
        c_i = {row['DisplayID']: float(row['Capacity']) for (_, row) in cap_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Capacity in {capacity_path}: {e}')
    try:
        v_j = {row['ProductName']: float(row['Value']) for (_, row) in prod_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Value in {products_path}: {e}')
    try:
        w_j = {row['ProductName']: float(row['Weight']) for (_, row) in prod_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Weight in {products_path}: {e}')
    if set(I) != set(c_i.keys()):
        raise ValueError('Mismatch between display area IDs and capacity keys.')
    if set(J) != set(v_j.keys()) or set(J) != set(w_j.keys()):
        raise ValueError('Mismatch between product names and value/weight keys.')
    m = gp.Model('Boat_Display_Allocation')
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()