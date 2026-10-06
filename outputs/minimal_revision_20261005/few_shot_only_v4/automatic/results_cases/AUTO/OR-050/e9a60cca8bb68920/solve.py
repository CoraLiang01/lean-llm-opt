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
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    if not {'ShelfID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    S = cap_df['ShelfID'].astype(str).tolist()
    P = prod_df['ProductName'].astype(str).tolist()
    c_s = cap_df.set_index('ShelfID')['Capacity'].to_dict()
    v_p = prod_df.set_index('ProductName')['Value'].to_dict()
    w_p = prod_df.set_index('ProductName')['Weight'].to_dict()
    for s in S:
        if s not in c_s:
            raise ValueError(f'Missing capacity for shelf {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    first_product_row = prod_df.iloc[0]
    p_star = str(first_product_row['ProductName'])
    keys = [(s, p) for s in S for p in P]
    m = gp.Model('Retail_Product_Shelf_Allocation')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[s, p] for (s, p) in keys)), GRB.MAXIMIZE)
    for s in S:
        m.addConstr(gp.quicksum((w_p[p] * x[s, p] for p in P)) <= c_s[s], name=f'cap_{s}')
    m.addConstr(gp.quicksum((x[s, p_star] for s in S)) >= 5, name='min_first_product')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')