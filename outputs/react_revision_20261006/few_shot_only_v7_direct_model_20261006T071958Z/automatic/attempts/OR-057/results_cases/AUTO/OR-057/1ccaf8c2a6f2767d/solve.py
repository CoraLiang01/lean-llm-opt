import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in csv_encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    platforms = cap_df['PlatformID'].tolist()
    genres = prod_df['ProductName'].tolist()
    try:
        capacity = cap_df.set_index('PlatformID')['Capacity'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    try:
        value = prod_df.set_index('ProductName')['Value'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Value to float: {e}')
    try:
        weight = prod_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Weight to float: {e}')
    for i in platforms:
        if i not in capacity:
            raise ValueError(f'Missing capacity for platform {i}')
    for j in genres:
        if j not in value:
            raise ValueError(f'Missing value for genre {j}')
        if j not in weight:
            raise ValueError(f'Missing weight for genre {j}')
    keys = [(i, j) for i in platforms for j in genres]
    m = gp.Model('VideoGameStore_Allocation')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * quantity_vars[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * quantity_vars[i, j] for j in genres)) <= capacity[i] for i in platforms), name='')
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