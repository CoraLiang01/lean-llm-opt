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
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
    products_df = read_csv_with_encodings(products_path)
    capacity_df = read_csv_with_encodings(capacity_path)
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f'Missing column {col} in products.csv')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('Missing column Capacity in capacity.csv')
    products_df = products_df.copy()
    products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
    products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
    product_keys = list(products_df['ProductName'])
    value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must contain exactly one row for total capacity.')
    try:
        total_capacity = float(capacity_df.iloc[0]['Capacity'])
    except Exception:
        raise ValueError('Capacity value in capacity.csv is not a valid number.')
    for k in product_keys:
        if k not in value_dict or k not in weight_dict:
            raise ValueError(f'Missing value or weight for product {k}')
    m = gp.Model('Pharmacy_Replenishment')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(product_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[i] * quantity_vars[i] for i in product_keys)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * quantity_vars[i] for i in product_keys)) <= total_capacity, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()