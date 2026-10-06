import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in csv_encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv'
    capacity_df = read_csv_with_encodings(capacity_path)
    products_df = read_csv_with_encodings(products_path)
    if not {'ShelfID', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    shelves = capacity_df['ShelfID'].astype(str).unique().tolist()
    products = products_df['ProductName'].astype(str).unique().tolist()
    capacity = {}
    for (_, row) in capacity_df.iterrows():
        shelf = str(row['ShelfID'])
        if shelf in capacity:
            raise ValueError(f'Duplicate ShelfID {shelf} in capacity.csv')
        capacity[shelf] = row['Capacity']
    value = {}
    weight = {}
    for (_, row) in products_df.iterrows():
        pname = str(row['ProductName'])
        if pname in value:
            raise ValueError(f'Duplicate ProductName {pname} in products.csv')
        value[pname] = row['Value']
        weight[pname] = row['Weight']
    if products_df.shape[0] == 0:
        raise ValueError('products.csv is empty.')
    first_product = str(products_df.iloc[0]['ProductName'])
    if first_product.casefold() != 'smartphone'.casefold():
        pass
    keys = [(s, p) for s in shelves for p in products]
    m = gp.Model('Retail_Product_Shelf_Allocation')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
    for s in shelves:
        m.addConstr(gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s], name=f'cap_{s}')
    m.addConstr(gp.quicksum((x[s, first_product] for s in shelves)) >= 5, name='min_smartphone')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()