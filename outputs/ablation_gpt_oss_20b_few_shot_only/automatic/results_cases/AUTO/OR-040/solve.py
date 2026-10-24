import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
    capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('Capacity column missing in capacity.csv')
    if capacity_df.empty:
        raise ValueError('capacity.csv is empty')
    try:
        cap_value = int(capacity_df['Capacity'].iloc[0])
    except Exception as e:
        raise ValueError('Capacity value is not an integer') from e
    products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Value'}
    if not required_cols.issubset(products_df.columns):
        missing = required_cols - set(products_df.columns)
        raise ValueError(f'Missing columns in products.csv: {missing}')
    if products_df['ProductName'].isnull().any():
        raise ValueError('Missing product names in products.csv')
    try:
        products_df['Value'] = products_df['Value'].astype(int)
    except Exception as e:
        raise ValueError('Value column contains non-integer entries') from e
    product_names = products_df['ProductName'].tolist()
    if len(product_names) != len(set(product_names)):
        raise ValueError('Duplicate product names found in products.csv')
    value_dict = dict(zip(product_names, products_df['Value']))
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value_dict[name] * x[name] for name in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((x[name] for name in product_names)) <= cap_value, name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')