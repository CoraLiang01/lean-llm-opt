import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity = int(capacity_df.iloc[0]['Capacity'])
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(product_names, products_df['Value']))
weight_dict = dict(zip(product_names, products_df['Weight']))
m = gp.Model('PharmaRestock')
var_names = {name: name.replace(' ', '_') for name in product_names}
x = m.addVars(product_names, vtype=gp.GRB.INTEGER, lb=0)
objective = gp.quicksum((value_dict[p] * x[p] for p in product_names))
m.setObjective(objective, gp.GRB.MAXIMIZE)
capacity_constraint = gp.quicksum((weight_dict[p] * x[p] for p in product_names))
m.addConstr(capacity_constraint <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal}')
    print('Daily order quantities:')
    for p in product_names:
        qty = x[p].X
        if qty > 0:
            print(f'  {p}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')