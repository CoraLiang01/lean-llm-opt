import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', sep=',')
products_df['Weight'] = products_df['Weight'].astype(int)
products_df['Value'] = products_df['Value'].astype(int)
m = gp.Model('SupermarketRestock')
x = m.addVars(products_df['ProductName'], vtype=gp.GRB.INTEGER)
obj = gp.quicksum((products_df.loc[i, 'Value'] * x[products_df.loc[i, 'ProductName']] for i in range(len(products_df))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
capacity_constr = gp.quicksum((products_df.loc[i, 'Weight'] * x[products_df.loc[i, 'ProductName']] for i in range(len(products_df)))) <= capacity
m.addConstr(capacity_constr)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal}')
    print('Order quantities:')
    for prod in products_df['ProductName']:
        qty = x[prod].X
        if qty > 0:
            print(f'  {prod}: {int(qty)}')
else:
    print(f'No optimal solution found. Status: {m.status}')