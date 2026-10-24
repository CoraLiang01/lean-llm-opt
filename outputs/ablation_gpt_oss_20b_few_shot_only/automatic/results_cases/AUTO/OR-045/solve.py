import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
    capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('Capacity column missing in capacity file')
    if capacity_df.shape[0] != 1:
        raise ValueError('Capacity file must contain exactly one row')
    try:
        capacity = int(capacity_df['Capacity'].iloc[0])
    except ValueError:
        raise ValueError('Capacity value must be an integer')
    if capacity < 0:
        raise ValueError('Capacity must be non‑negative')
    products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Weight', 'Value'}
    if not required_cols.issubset(products_df.columns):
        missing = required_cols - set(products_df.columns)
        raise ValueError(f'Missing columns in products file: {missing}')
    try:
        products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
        products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
    except Exception as e:
        raise ValueError(f'Error converting numeric columns: {e}')
    product_names = products_df['ProductName'].tolist()
    weight = dict(zip(product_names, products_df['Weight']))
    value = dict(zip(product_names, products_df['Value']))
    for pname in product_names:
        if pname not in weight or pname not in value:
            raise ValueError(f'Missing weight or value for product {pname}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight[p] * x[p] for p in product_names)) <= capacity, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()