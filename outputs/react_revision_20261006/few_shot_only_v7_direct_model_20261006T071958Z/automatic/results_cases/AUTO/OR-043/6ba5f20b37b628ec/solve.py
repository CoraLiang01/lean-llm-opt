import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            products_df = pd.read_csv(product_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {product_path} with tried encodings.')
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
        raise ValueError(f'Missing columns in products.csv: {required_product_cols - set(products_df.columns)}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    I = []
    v = {}
    w = {}
    for (idx, row) in products_df.iterrows():
        prod = row['ProductName']
        if prod in I:
            continue
        try:
            v_i = float(row['Value'])
            w_i = float(row['Weight'])
        except Exception:
            raise ValueError(f'Non-numeric Value or Weight for product {prod}')
        I.append(prod)
        v[prod] = v_i
        w[prod] = w_i
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must have exactly one record for total capacity.')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception:
        raise ValueError('Non-numeric Capacity in capacity.csv')
    m = gp.Model('PharmacyDrugOrder')
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * quantity_vars[i] for i in I)) <= C, name='stock_capacity')
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