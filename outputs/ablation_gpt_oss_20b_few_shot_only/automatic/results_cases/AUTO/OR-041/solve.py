import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
    df_cap = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in df_cap.columns or df_cap.empty:
        raise ValueError('Capacity column missing or file empty')
    try:
        C = int(df_cap['Capacity'].iloc[0])
    except Exception as e:
        raise ValueError('Capacity value invalid') from e
    df_prod = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = ['ProductName', 'Value', 'Weight']
    for col in required_cols:
        if col not in df_prod.columns:
            raise ValueError(f'Missing column {col} in products file')
    if df_prod.empty:
        raise ValueError('Products file is empty')
    try:
        df_prod['Value'] = df_prod['Value'].astype(int)
        df_prod['Weight'] = df_prod['Weight'].astype(int)
    except Exception as e:
        raise ValueError('Non-numeric Value or Weight') from e
    product_names = df_prod['ProductName'].tolist()
    v = dict(zip(product_names, df_prod['Value']))
    w = dict(zip(product_names, df_prod['Weight']))
    for name in product_names:
        if name not in v or name not in w:
            raise ValueError(f'Missing parameters for product {name}')
    model = gurobipy.Model('RealEstateDev')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((v[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((w[i] * x[i] for i in product_names)) <= C)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()