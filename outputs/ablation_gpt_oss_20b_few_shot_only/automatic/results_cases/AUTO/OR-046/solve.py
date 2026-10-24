import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
    cap_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in cap_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    if cap_df.empty:
        raise ValueError('capacity.csv is empty')
    try:
        C = float(cap_df.loc[0, 'Capacity'])
    except Exception as exc:
        raise ValueError(f"Invalid capacity value: {cap_df.loc[0, 'Capacity']}") from exc
    prod_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = ['ProductName', 'Weight', 'Value']
    for col in required_cols:
        if col not in prod_df.columns:
            raise ValueError(f"Missing column '{col}' in products.csv")
    prod_df['ProductName'] = prod_df['ProductName'].str.strip()
    try:
        prod_df['Weight'] = pd.to_numeric(prod_df['Weight'], errors='raise')
        prod_df['Value'] = pd.to_numeric(prod_df['Value'], errors='raise')
    except Exception as exc:
        raise ValueError('Non-numeric weight or value in products.csv') from exc
    if prod_df['ProductName'].duplicated().any():
        dup = prod_df['ProductName'][prod_df['ProductName'].duplicated()].unique()
        raise ValueError(f'Duplicate product names: {dup}')
    I = prod_df['ProductName'].tolist()
    weight = dict(zip(I, prod_df['Weight']))
    value = dict(zip(I, prod_df['Value']))
    for i in I:
        if i not in weight or i not in value:
            raise ValueError(f'Missing weight or value for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weight[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')