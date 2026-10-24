import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
    df_cap = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'StorageID' not in df_cap.columns or 'Capacity' not in df_cap.columns:
        raise ValueError('Capacity CSV missing required columns')
    df_cap['StorageID'] = df_cap['StorageID'].str.strip()
    df_cap['Capacity'] = df_cap['Capacity'].str.strip()
    try:
        df_cap['Capacity'] = df_cap['Capacity'].astype(int)
    except ValueError as e:
        raise ValueError('Capacity values must be integers') from e
    C = dict(zip(df_cap['StorageID'], df_cap['Capacity']))
    df_prod = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = ['ProductName', 'Value', 'Weight']
    if not all((col in df_prod.columns for col in required_cols)):
        raise ValueError('Products CSV missing required columns')
    df_prod['ProductName'] = df_prod['ProductName'].str.strip()
    df_prod['Value'] = df_prod['Value'].str.strip()
    df_prod['Weight'] = df_prod['Weight'].str.strip()
    try:
        df_prod['Value'] = df_prod['Value'].astype(int)
    except ValueError as e:
        raise ValueError('Product Value must be integers') from e
    try:
        df_prod['Weight'] = df_prod['Weight'].astype(int)
    except ValueError as e:
        raise ValueError('Product Weight must be integers') from e
    V = dict(zip(df_prod['ProductName'], df_prod['Value']))
    W = dict(zip(df_prod['ProductName'], df_prod['Weight']))
    I = list(C.keys())
    J = list(V.keys())
    if set(J) != set(W.keys()):
        raise ValueError('Product names mismatch between Value and Weight')
    if not I:
        raise ValueError('No storage areas defined')
    if not J:
        raise ValueError('No products defined')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((V[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((W[j] * x[i, j] for j in J)) <= C[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()