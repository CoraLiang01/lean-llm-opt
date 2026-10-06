import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            products = pd.read_csv(product_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {product_path} with supported encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
    for encoding in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with supported encodings.')
    required_product_cols = {'ProductName', 'Value', 'Weight'}
    if not required_product_cols.issubset(products.columns):
        raise ValueError(f'products.csv missing columns: {required_product_cols - set(products.columns)}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("capacity.csv missing 'Capacity' column.")
    grouped = products.groupby('ProductName', as_index=False).agg({'Value': 'sum', 'Weight': 'sum'})
    items = grouped['ProductName'].tolist()
    profit = dict(zip(grouped['ProductName'], grouped['Value']))
    weight = dict(zip(grouped['ProductName'], grouped['Weight']))
    total_capacity = capacity_df['Capacity'].sum()
    if any((i not in profit or i not in weight for i in items)):
        raise ValueError('Missing profit or weight for some items.')
    m = gp.Model('CarSales2')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((profit[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in items)) <= total_capacity, name='capacity')
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