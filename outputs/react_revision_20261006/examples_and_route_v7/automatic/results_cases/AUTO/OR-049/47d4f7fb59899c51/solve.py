import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv', sep=',', dtype=str, keep_default_na=False)
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
product_names = products_df['ProductName'].astype(str).tolist()
capacity_dict = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(float)))
value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(float)))
if set(shelf_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between shelf_ids and capacity_dict keys.')
if set(product_names) != set(value_dict.keys()) or set(product_names) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product_names and value/weight dict keys.')
decision_keys = [(i, j) for i in shelf_ids for j in product_names]
m = gp.Model('RetailShelfAllocation')
x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for (i, j) in decision_keys)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in decision_keys:
        var = x_vars[i, j]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')