import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv'
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
    if len(products) == 0:
        raise ValueError('No products found in products.csv.')
    first_product = products[0]
    keys = [(i, j) for i in shelves for j in products]
    m = gp.Model('Retail_Display_Allocation')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * quantity_vars[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
    for i in shelves:
        m.addConstr(gp.quicksum((weight[j] * quantity_vars[i, j] for j in products)) <= capacity[i], name=f'cap_{i}')
    m.addConstr(gp.quicksum((quantity_vars[i, first_product] for i in shelves)) >= 5, name='min_first_product')
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