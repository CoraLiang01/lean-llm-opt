import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
    try:
        products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding='latin-1')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
    try:
        capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding='latin-1')
    if 'ProductName' not in products_df.columns:
        raise ValueError("Missing 'ProductName' column in products.csv")
    I = products_df['ProductName'].tolist()
    if 'Value' not in products_df.columns:
        raise ValueError("Missing 'Value' column in products.csv")
    if 'Weight' not in products_df.columns:
        raise ValueError("Missing 'Weight' column in products.csv")
    try:
        v_i = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
        w_i = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Value/Weight to float: {e}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    for i in I:
        if i not in v_i or i not in w_i:
            raise ValueError(f"Missing Value or Weight for area '{i}'")
    m = gp.Model('NYC_Development')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w_i[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()