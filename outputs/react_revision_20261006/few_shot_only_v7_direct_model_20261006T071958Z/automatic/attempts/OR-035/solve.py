import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {products_path} with tried encodings.')
    for encoding in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {capacity_path} with tried encodings.')
    required_product_cols = {'ProductName', 'Value', 'Weight'}
    if not required_product_cols.issubset(products_df.columns):
        raise ValueError(f'products.csv missing columns: {required_product_cols - set(products_df.columns)}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('capacity.csv missing column: Capacity')
    I = products_df['ProductName'].tolist()
    if len(I) != len(set(I)):
        raise ValueError('Duplicate ProductName entries found in products.csv.')
    try:
        v_i = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
        w_i = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Value/Weight to float: {e}')
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must have exactly one record for total capacity.')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    for i in I:
        if i not in v_i or i not in w_i:
            raise ValueError(f'Missing Value or Weight for product {i}')
    m = gp.Model('Bakery_Bread_Stocking')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w_i[i] * quantity_vars[i] for i in I)) <= C, name='storage_capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()