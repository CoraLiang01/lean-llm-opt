import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv']
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_fallback(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_df = read_csv_with_fallback(csv_paths[0])
    products_df = read_csv_with_fallback(csv_paths[1])
    if not {'ShelfID', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    shelves = capacity_df['ShelfID'].tolist()
    products = products_df['ProductName'].tolist()
    try:
        capacity = {row['ShelfID']: float(row['Capacity']) for (_, row) in capacity_df.iterrows()}
    except Exception:
        raise ValueError('Non-numeric or missing Capacity in capacity.csv.')
    try:
        value = {row['ProductName']: float(row['Value']) for (_, row) in products_df.iterrows()}
        weight = {row['ProductName']: float(row['Weight']) for (_, row) in products_df.iterrows()}
    except Exception:
        raise ValueError('Non-numeric or missing Value/Weight in products.csv.')
    for s in shelves:
        if s not in capacity:
            raise ValueError(f'Missing capacity for shelf {s}.')
    for p in products:
        if p not in value or p not in weight:
            raise ValueError(f'Missing value/weight for product {p}.')
    keys = [(s, p) for s in shelves for p in products]
    m = gp.Model('Retail_Shelf_Allocation')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[p] * quantity_vars[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    for s in shelves:
        m.addConstr(gp.quicksum((weight[p] * quantity_vars[s, p] for p in products)) <= capacity[s], name=f'cap_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()