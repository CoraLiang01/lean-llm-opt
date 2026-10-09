import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
    for enc in csv_encodings:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for enc in csv_encodings:
        try:
            products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    if not {'SectionID', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    sections = capacity_df['SectionID'].tolist()
    products = products_df['ProductName'].tolist()
    try:
        capacity_dict = capacity_df.set_index('SectionID')['Capacity'].astype(float).to_dict()
    except Exception:
        raise ValueError('Non-numeric or missing Capacity in capacity.csv.')
    try:
        value_dict = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
        weight_dict = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception:
        raise ValueError('Non-numeric or missing Value/Weight in products.csv.')
    for s in sections:
        if s not in capacity_dict:
            raise ValueError(f'Missing capacity for section {s}.')
    for p in products:
        if p not in value_dict or p not in weight_dict:
            raise ValueError(f'Missing value/weight for product {p}.')
    keys = [(i, j) for i in sections for j in products]
    m = gp.Model('Supermarket_Section_Allocation')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[j] * quantity_vars[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight_dict[j] * quantity_vars[i, j] for j in products)) <= capacity_dict[i] for i in sections), name='')
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