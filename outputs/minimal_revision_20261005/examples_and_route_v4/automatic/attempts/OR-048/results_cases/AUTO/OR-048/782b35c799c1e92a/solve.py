import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
    df_capacity = pd.read_csv(capacity_path, sep=',')
    df_products = pd.read_csv(products_path, sep=',')
    storage_ids = df_capacity['StorageID'].astype(int).tolist()
    product_names = df_products['ProductName'].astype(str).tolist()
    capacity = dict(zip(df_capacity['StorageID'].astype(int), df_capacity['Capacity'].astype(int)))
    value = dict(zip(df_products['ProductName'].astype(str), df_products['Value'].astype(int)))
    weight = dict(zip(df_products['ProductName'].astype(str), df_products['Weight'].astype(int)))
    if set(storage_ids) != set(capacity.keys()):
        raise ValueError('Mismatch between storage_ids and capacity keys.')
    if set(product_names) != set(value.keys()) or set(product_names) != set(weight.keys()):
        raise ValueError('Mismatch between product_names and value/weight keys.')
    keys = [(i, j) for i in storage_ids for j in product_names]
    m = gp.Model('Amazon_AC_Storage')
    x = m.addVars(keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[j] * x[i, j] for i in storage_ids for j in product_names)), gp.GRB.MAXIMIZE)
    for i in storage_ids:
        m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in product_names)) <= capacity[i], name=f'cap_{i}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')