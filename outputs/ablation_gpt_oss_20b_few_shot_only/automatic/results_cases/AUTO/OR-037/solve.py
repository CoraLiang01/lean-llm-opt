import pandas as pd
import gurobipy
from gurobipy import GRB
CAPACITY_PATH = 'Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
PRODUCTS_PATH = 'Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'

def solve_problem():
    df_capacity = pd.read_csv(CAPACITY_PATH, dtype=str, keep_default_na=False)
    if 'Capacity' not in df_capacity.columns or df_capacity.empty:
        raise ValueError('Capacity column missing or file empty')
    capacity_str = df_capacity['Capacity'].iloc[0].strip()
    if capacity_str == '':
        raise ValueError('Capacity value is missing')
    try:
        capacity = int(capacity_str)
    except ValueError:
        raise ValueError(f"Capacity value '{capacity_str}' is not an integer")
    df_products = pd.read_csv(PRODUCTS_PATH, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(df_products.columns):
        missing = required_cols - set(df_products.columns)
        raise ValueError(f'Missing columns in products file: {missing}')
    product_names = []
    value_dict = {}
    weight_dict = {}
    for (_, row) in df_products.iterrows():
        pname = row['ProductName'].strip()
        if pname == '':
            raise ValueError('ProductName is empty')
        if pname in product_names:
            raise ValueError(f'Duplicate product name: {pname}')
        val_str = row['Value'].strip()
        wgt_str = row['Weight'].strip()
        if val_str == '' or wgt_str == '':
            raise ValueError(f"Missing Value or Weight for product '{pname}'")
        try:
            val = int(val_str)
            wgt = int(wgt_str)
        except ValueError:
            raise ValueError(f"Non-integer Value or Weight for product '{pname}'")
        product_names.append(pname)
        value_dict[pname] = val
        weight_dict[pname] = wgt
    if not product_names:
        raise ValueError('No products found in products file')
    m = gurobipy.Model('CarSalesInventory')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value_dict[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight_dict[i] * x[i] for i in product_names)) <= capacity)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()