import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
    capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('Capacity column missing in capacity.csv')
    if capacity_df.shape[0] != 1:
        raise ValueError('capacity.csv should contain exactly one row')
    capacity_str = capacity_df.iloc[0]['Capacity'].strip()
    if capacity_str == '':
        raise ValueError('Capacity value is missing')
    try:
        C = int(capacity_str)
    except ValueError:
        raise ValueError(f'Capacity value "{capacity_str}" is not an integer')
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
    prod_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = ['ProductName', 'Value', 'Weight']
    for col in required_cols:
        if col not in prod_df.columns:
            raise ValueError(f'{col} column missing in products.csv')
    product_names = []
    profit = {}
    weight = {}
    for (_, row) in prod_df.iterrows():
        name = row['ProductName'].strip()
        if name == '':
            continue
        if name in product_names:
            raise ValueError(f'Duplicate product name "{name}" in products.csv')
        product_names.append(name)
        val_str = row['Value'].strip()
        if val_str == '':
            raise ValueError(f'Value missing for product "{name}"')
        try:
            profit_val = int(val_str)
        except ValueError:
            raise ValueError(f'Value "{val_str}" for product "{name}" is not an integer')
        weight_str = row['Weight'].strip()
        if weight_str == '':
            raise ValueError(f'Weight missing for product "{name}"')
        try:
            weight_val = int(weight_str)
        except ValueError:
            raise ValueError(f'Weight "{weight_str}" for product "{name}" is not an integer')
        profit[name] = profit_val
        weight[name] = weight_val
    if not product_names:
        raise ValueError('No products found in products.csv')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((profit[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight[i] * x[i] for i in product_names)) <= C, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()