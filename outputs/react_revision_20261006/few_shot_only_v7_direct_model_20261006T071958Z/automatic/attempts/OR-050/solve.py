import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv'
    cap_df = read_csv_with_encodings(capacity_path, dtype=str, keep_default_na=False)
    prod_df = read_csv_with_encodings(products_path, dtype=str, keep_default_na=False)
    I = cap_df['ShelfID'].tolist()
    J = prod_df['ProductName'].tolist()
    try:
        c_i = cap_df.set_index('ShelfID')['Capacity'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError(f'Error converting Capacity to float: {e}')
    try:
        v_j = prod_df.set_index('ProductName')['Value'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError(f'Error converting Value to float: {e}')
    try:
        w_j = prod_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError(f'Error converting Weight to float: {e}')
    if len(prod_df) == 0:
        raise RuntimeError('products.csv is empty.')
    j_star = prod_df.iloc[0]['ProductName']
    for i in I:
        if i not in c_i:
            raise RuntimeError(f'Missing capacity for ShelfID {i}')
    for j in J:
        if j not in v_j:
            raise RuntimeError(f'Missing value for ProductName {j}')
        if j not in w_j:
            raise RuntimeError(f'Missing weight for ProductName {j}')
    m = gp.Model('retail_display_allocation')
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.addConstr(gp.quicksum((quantity_vars[i, j_star] for i in I)) >= 5, name='min_quantity_first_product')
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