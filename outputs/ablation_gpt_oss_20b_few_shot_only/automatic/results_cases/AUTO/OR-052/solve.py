import gurobipy
from gurobipy import GRB
import pandas as pd
CAPACITY_PATH = 'Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
PRODUCTS_PATH = 'Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'

def solve_problem():
    capacity_df = pd.read_csv(CAPACITY_PATH, dtype=str, keep_default_na=False)
    if 'BookshelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
        raise ValueError('Capacity CSV missing required columns')
    capacity_df['Capacity'] = pd.to_numeric(capacity_df['Capacity'], errors='raise')
    I = list(capacity_df['BookshelfID'])
    c_i = dict(zip(capacity_df['BookshelfID'], capacity_df['Capacity']))
    products_df = pd.read_csv(PRODUCTS_PATH, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(products_df.columns):
        raise ValueError('Products CSV missing required columns')
    products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
    products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
    J = list(products_df['ProductName'])
    v_j = dict(zip(products_df['ProductName'], products_df['Value']))
    w_j = dict(zip(products_df['ProductName'], products_df['Weight']))
    for i in I:
        if i not in c_i:
            raise ValueError(f'Missing capacity for bookshelf {i}')
    for j in J:
        if j not in v_j or j not in w_j:
            raise ValueError(f'Missing value or weight for product {j}')
    m = gurobipy.Model('BookShelfAllocation')
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v_j[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((w_j[j] * x[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()