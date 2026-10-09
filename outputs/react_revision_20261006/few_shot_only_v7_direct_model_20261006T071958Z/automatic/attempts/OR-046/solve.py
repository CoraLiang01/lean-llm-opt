import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            products_df = pd.read_csv(product_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {product_path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
    for encoding in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {capacity_path} with tried encodings.')
    if 'ProductName' not in products_df.columns or 'Weight' not in products_df.columns or 'Value' not in products_df.columns:
        raise ValueError('products.csv must contain columns: ProductName, Weight, Value')
    I = products_df['ProductName'].tolist()
    try:
        w_i = pd.Series(products_df['Weight'].astype(float).values, index=products_df['ProductName']).to_dict()
        v_i = pd.Series(products_df['Value'].astype(float).values, index=products_df['ProductName']).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Weight/Value to float: {e}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('capacity.csv must contain column: Capacity')
    try:
        C = float(capacity_df['Capacity'].iloc[0])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    missing_w = [i for i in I if i not in w_i]
    missing_v = [i for i in I if i not in v_i]
    if missing_w or missing_v:
        raise ValueError(f'Missing weights for: {missing_w}, or values for: {missing_v}')
    m = gp.Model('SupermarketStockReplenishment')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w_i[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')