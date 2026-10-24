import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
    cap_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in cap_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    if cap_df.empty:
        raise ValueError('capacity.csv is empty')
    try:
        capacity = float(cap_df['Capacity'].iloc[0])
    except ValueError:
        raise ValueError('Capacity value is not numeric')
    prod_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = ['ProductName', 'Value', 'Weight']
    for col in required_cols:
        if col not in prod_df.columns:
            raise ValueError(f"Missing '{col}' column in products.csv")
    if prod_df.empty:
        raise ValueError('products.csv is empty')
    product_names = prod_df['ProductName'].tolist()
    try:
        values = prod_df['Value'].astype(float).tolist()
    except ValueError:
        raise ValueError("Non-numeric value found in 'Value' column")
    try:
        weights = prod_df['Weight'].astype(float).tolist()
    except ValueError:
        raise ValueError("Non-numeric value found in 'Weight' column")
    value_dict = dict(zip(product_names, values))
    weight_dict = dict(zip(product_names, weights))
    for pname in product_names:
        if pname not in value_dict or pname not in weight_dict:
            raise ValueError(f"Missing parameters for product '{pname}'")
    m = gurobipy.Model('CarSalesInventory')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value_dict[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight_dict[i] * x[i] for i in product_names)) <= capacity, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()