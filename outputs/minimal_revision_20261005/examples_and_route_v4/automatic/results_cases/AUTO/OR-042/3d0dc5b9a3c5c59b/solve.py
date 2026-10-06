import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv'
    df_products = pd.read_csv(products_path, sep=',')
    df_capacity = pd.read_csv(capacity_path, sep=',')
    if df_products['ProductName'].isnull().any():
        raise ValueError('Missing ProductName in products.csv')
    product_ids = df_products['ProductName'].astype(str).tolist()
    if df_products['Value'].isnull().any():
        raise ValueError('Missing Value in products.csv')
    if df_products['Weight'].isnull().any():
        raise ValueError('Missing Weight in products.csv')
    value = dict(zip(product_ids, df_products['Value'].astype(int)))
    weight = dict(zip(product_ids, df_products['Weight'].astype(int)))
    if set(value.keys()) != set(product_ids) or set(weight.keys()) != set(product_ids):
        raise ValueError('Mismatch in product keys for value/weight')
    if df_capacity.shape[0] != 1 or 'Capacity' not in df_capacity.columns:
        raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column")
    capacity = int(df_capacity['Capacity'].iloc[0])
    m = gp.Model('PharmacyDrugOrder')
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_ids)) <= capacity, name='cap')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in product_ids:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()