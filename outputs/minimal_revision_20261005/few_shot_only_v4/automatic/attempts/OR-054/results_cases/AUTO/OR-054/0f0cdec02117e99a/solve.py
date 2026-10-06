import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            products_df = pd.read_csv(products_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    if not {'ShelfID', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    shelves = capacity_df['ShelfID'].astype(str).unique().tolist()
    products = products_df['ProductName'].astype(str).unique().tolist()
    capacity_dict = {}
    for (_, row) in capacity_df.iterrows():
        key = str(row['ShelfID'])
        val = row['Capacity']
        if key in capacity_dict:
            capacity_dict[key] += val
        else:
            capacity_dict[key] = val
    value_dict = {}
    weight_dict = {}
    for (_, row) in products_df.iterrows():
        key = str(row['ProductName'])
        v = row['Value']
        w = row['Weight']
        if key in value_dict:
            value_dict[key] += v
            weight_dict[key] += w
        else:
            value_dict[key] = v
            weight_dict[key] = w
    if set(shelves) != set(capacity_dict.keys()):
        raise ValueError('Mismatch between shelves and capacity_dict keys.')
    if set(products) != set(value_dict.keys()) or set(products) != set(weight_dict.keys()):
        raise ValueError('Mismatch between products and value/weight dict keys.')
    m = gp.Model('BigMart_Shelf_Allocation')
    m.setParam('MIPGap', 0.0001)
    keys = [(s, p) for s in shelves for p in products]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    for s in shelves:
        m.addConstr(gp.quicksum((weight_dict[p] * x[s, p] for p in products)) <= capacity_dict[s], name=f'cap_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()