import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
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
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
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
        raise ValueError('capacity.csv must contain columns: ShelfID, Capacity')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
    shelves = list(shelves_df['ShelfID'])
    products = list(products_df['ProductName'])
    if len(set(shelves)) != len(shelves):
        raise ValueError('Duplicate ShelfID found in capacity.csv')
    if len(set(products)) != len(products):
        raise ValueError('Duplicate ProductName found in products.csv')
    capacity = dict(zip(shelves_df['ShelfID'], shelves_df['Capacity']))
    value = dict(zip(products_df['ProductName'], products_df['Value']))
    weight = dict(zip(products_df['ProductName'], products_df['Weight']))
    for s in shelves:
        if pd.isnull(capacity[s]):
            raise ValueError(f'Missing capacity for shelf {s}')
    for p in products:
        if pd.isnull(value[p]) or pd.isnull(weight[p]):
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('BigMart_Shelf_Allocation')
    m.Params.MIPGap = 0.0001
    keys = [(s, p) for s in shelves for p in products]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s] for s in shelves), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()