import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    if not {'ShelfID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    cap_df = cap_df.drop_duplicates(subset=['ShelfID'])
    prod_df = prod_df.drop_duplicates(subset=['ProductName'])
    shelves = cap_df['ShelfID'].tolist()
    products = prod_df['ProductName'].tolist()
    capacity = dict(zip(cap_df['ShelfID'], cap_df['Capacity']))
    value = dict(zip(prod_df['ProductName'], prod_df['Value']))
    weight = dict(zip(prod_df['ProductName'], prod_df['Weight']))
    for s in shelves:
        if s not in capacity:
            raise ValueError(f'Missing capacity for shelf {s}')
    for p in products:
        if p not in value or p not in weight:
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('BigMart_Shelf_Allocation')
    m.Params.MIPGap = 0.0001
    x_keys = [(s, p) for s in shelves for p in products]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    for s in shelves:
        m.addConstr(gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s])
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()