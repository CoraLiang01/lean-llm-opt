import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv', sep=',', dtype=str, keep_default_na=False)
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
product_ids = products_df['ProductName'].astype(int).tolist()
capacity = {}
for (idx, row) in capacity_df.iterrows():
    sid = int(row['ShelfID'])
    cap = int(row['Capacity'])
    if sid in capacity:
        raise ValueError(f'Duplicate ShelfID {sid} in capacity.csv')
    capacity[sid] = cap
value = {}
weight = {}
for (idx, row) in products_df.iterrows():
    pid = int(row['ProductName'])
    val = int(row['Value'])
    wgt = int(row['Weight'])
    if pid in value or pid in weight:
        raise ValueError(f'Duplicate ProductName {pid} in products.csv')
    value[pid] = val
    weight[pid] = wgt
if set(shelf_ids) != set(capacity.keys()):
    raise ValueError('Mismatch between shelf_ids and capacity keys')
if set(product_ids) != set(value.keys()) or set(product_ids) != set(weight.keys()):
    raise ValueError('Mismatch between product_ids and value/weight keys')
x_keys = [(i, j) for i in shelf_ids for j in product_ids]
m = gp.Model('BigMartShelfAllocation')
x_vars = m.addVars(x_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight[j] * x_vars[i, j] for j in product_ids)) <= capacity[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in shelf_ids:
        for j in product_ids:
            var = x_vars[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')