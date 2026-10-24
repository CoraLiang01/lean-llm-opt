import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv'
    capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    capacity_df['PlatformId'] = capacity_df['PlatformId'].str.strip()
    capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
    capacity_df['Capacity'] = pd.to_numeric(capacity_df['Capacity'], errors='raise')
    capacity = dict(zip(capacity_df['PlatformId'], capacity_df['Capacity']))
    products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    products_df['ProductName'] = products_df['ProductName'].str.strip()
    products_df['Value'] = products_df['Value'].str.strip()
    products_df['Weight'] = products_df['Weight'].str.strip()
    products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
    products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
    value = dict(zip(products_df['ProductName'], products_df['Value']))
    weight = dict(zip(products_df['ProductName'], products_df['Weight']))
    I = set(capacity.keys())
    J = set(value.keys())
    if J != set(weight.keys()):
        missing = J.symmetric_difference(set(weight.keys()))
        raise ValueError(f'Missing weight or value for products: {missing}')
    for j in J:
        if j not in value or j not in weight:
            raise ValueError(f'Missing value or weight for product: {j}')
    for i in I:
        if i not in capacity:
            raise ValueError(f'Missing capacity for platform: {i}')
    m = gurobipy.Model('VideoGameListing')
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((weight[j] * x[i, j] for j in J)) <= capacity[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()