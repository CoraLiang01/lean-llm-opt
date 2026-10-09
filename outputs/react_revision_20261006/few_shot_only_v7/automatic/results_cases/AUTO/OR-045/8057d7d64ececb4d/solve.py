import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            products_df = pd.read_csv(product_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {product_path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
    for encoding in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for col in ['ProductName', 'Weight', 'Value']:
        if col not in products_df.columns:
            raise ValueError(f'Missing column {col} in products.csv')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('Missing column Capacity in capacity.csv')
    I = list(products_df['ProductName'])
    if len(I) != len(set(I)):
        raise ValueError('Duplicate ProductName entries found in products.csv')
    try:
        w = {row['ProductName']: float(row['Weight']) for (_, row) in products_df.iterrows()}
        v = {row['ProductName']: float(row['Value']) for (_, row) in products_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error converting Weight/Value to float: {e}')
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must contain exactly one row with the total Capacity.')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    m = gp.Model('SupermarketRestock')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')