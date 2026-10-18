import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv(products_path, sep=',')
product_names = products_df['ProductName'].tolist()
profits = products_df['Value'].tolist()
weights = products_df['Weight'].tolist()
m = gp.Model('CarSalesReplenishment')
n = len(product_names)
x = m.addVars(range(n), vtype=gp.GRB.INTEGER, lb=0)
prod_vars = {product_names[i]: x[i] for i in range(n)}
m.setObjective(gp.quicksum((profits[i] * x[i] for i in range(n))), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in range(n))) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_profit = m.objVal
    total_weight_used = sum((weights[i] * x[i].X for i in range(n)))
    print(f'Optimal total profit: {total_profit:.2f}')
    print(f'Total weight used: {total_weight_used:.2f} / {capacity}')
    print('\n--- Ordering Plan ---')
    for name in product_names:
        qty = prod_vars[name].X
        if qty > 0:
            print(f'{name:15s}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')