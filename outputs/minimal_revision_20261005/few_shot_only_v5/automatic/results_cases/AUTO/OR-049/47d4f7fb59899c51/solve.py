import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
    try:
        shelves_df = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            shelves_df = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                shelves_df = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                shelves_df = pd.read_csv(capacity_path, encoding='latin-1')
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
    try:
        products_df = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            products_df = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                products_df = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                products_df = pd.read_csv(products_path, encoding='latin-1')
    if not {'ShelfID', 'Capacity'}.issubset(shelves_df.columns):
        raise ValueError('capacity.csv missing required columns')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns')
    shelves = [str(s) for s in shelves_df['ShelfID']]
    products = [str(p) for p in products_df['ProductName']]
    capacity = {}
    for (_, row) in shelves_df.iterrows():
        key = str(row['ShelfID'])
        if key in capacity:
            raise ValueError(f'Duplicate ShelfID: {key}')
        capacity[key] = float(row['Capacity'])
    value = {}
    weight = {}
    for (_, row) in products_df.iterrows():
        key = str(row['ProductName'])
        if key in value or key in weight:
            raise ValueError(f'Duplicate ProductName: {key}')
        value[key] = float(row['Value'])
        weight[key] = float(row['Weight'])
    if set(shelves) != set(capacity.keys()):
        raise ValueError('Mismatch between shelves and capacity keys')
    if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
        raise ValueError('Mismatch between products and value/weight keys')
    m = gp.Model('Retail_Shelf_Allocation')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(s, p) for s in shelves for p in products]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    for s in shelves:
        m.addConstr(gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s], name=f'cap_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()