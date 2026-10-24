import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
    capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
        raise ValueError('Capacity CSV must contain ShelfID and Capacity columns')
    capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
    shelves = capacity_df['ShelfID'].tolist()
    capacity_dict = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
    products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(products_df.columns):
        raise ValueError('Products CSV must contain ProductName, Value, and Weight columns')
    products_df['Value'] = products_df['Value'].astype(float)
    products_df['Weight'] = products_df['Weight'].astype(float)
    products = products_df['ProductName'].tolist()
    value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
    if not set(shelves).issubset(capacity_dict.keys()):
        missing = set(shelves) - set(capacity_dict.keys())
        raise ValueError(f'Missing capacity for shelves: {missing}')
    if not set(products).issubset(value_dict.keys()):
        missing = set(products) - set(value_dict.keys())
        raise ValueError(f'Missing value for products: {missing}')
    if not set(products).issubset(weight_dict.keys()):
        missing = set(products) - set(weight_dict.keys())
        raise ValueError(f'Missing weight for products: {missing}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    var_keys = [(i, j) for i in shelves for j in products]
    x = model.addVars(var_keys, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
    for i in shelves:
        model.addConstr(gurobipy.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i])
    model.optimize()
    return model
m = solve_problem()