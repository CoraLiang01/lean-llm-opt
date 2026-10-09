import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv']
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_fallback(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    cap_df = read_csv_with_fallback(csv_paths[0])
    prod_df = read_csv_with_fallback(csv_paths[1])
    if not {'DisplayID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    display_ids = cap_df['DisplayID'].tolist()
    product_names = prod_df['ProductName'].tolist()
    if len(set(display_ids)) != len(display_ids):
        raise ValueError('Duplicate DisplayID in capacity.csv.')
    if len(set(product_names)) != len(product_names):
        raise ValueError('Duplicate ProductName in products.csv.')
    try:
        capacity = {row['DisplayID']: float(row['Capacity']) for (_, row) in cap_df.iterrows()}
    except Exception:
        raise ValueError('Non-numeric or missing Capacity in capacity.csv.')
    try:
        value = {row['ProductName']: float(row['Value']) for (_, row) in prod_df.iterrows()}
        weight = {row['ProductName']: float(row['Weight']) for (_, row) in prod_df.iterrows()}
    except Exception:
        raise ValueError('Non-numeric or missing Value/Weight in products.csv.')
    keys = [(i, j) for i in display_ids for j in product_names]
    m = gp.Model('Boat_Display_Allocation')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * quantity_vars[i, j] for i in display_ids for j in product_names)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * quantity_vars[i, j] for j in product_names)) <= capacity[i] for i in display_ids), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()