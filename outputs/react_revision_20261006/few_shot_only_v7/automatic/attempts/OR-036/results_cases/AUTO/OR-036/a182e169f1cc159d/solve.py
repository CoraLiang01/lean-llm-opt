import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            products_df = pd.read_csv(product_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {product_path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
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
        missing = required_product_cols - set(products_df.columns)
        raise ValueError(f'Missing columns in products.csv: {missing}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv.")
    products_grouped = products_df.groupby('ProductName', as_index=False).agg({'Value': lambda x: x.iloc[0], 'Weight': lambda x: x.iloc[0]})
    I = list(products_grouped['ProductName'])
    try:
        value = {row['ProductName']: float(row['Value']) for (_, row) in products_grouped.iterrows()}
        weight = {row['ProductName']: float(row['Weight']) for (_, row) in products_grouped.iterrows()}
    except Exception as e:
        raise ValueError(f'Error converting Value/Weight to float: {e}')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    for i in I:
        if i not in value or i not in weight:
            raise ValueError(f'Missing value or weight for product {i}')
    m = gp.Model('CarSales_Inventory')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()