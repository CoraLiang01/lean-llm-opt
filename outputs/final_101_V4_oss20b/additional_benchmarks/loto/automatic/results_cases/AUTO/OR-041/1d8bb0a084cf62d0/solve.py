import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
total_capacity = int(capacity_df.loc[0, 'Capacity'])
areas = products_df['ProductName'].tolist()
m = gp.Model('RealEstateDevelopment')
x = m.addVars(areas, vtype=gp.GRB.CONTINUOUS)
objective = gp.quicksum((products_df.loc[products_df['ProductName'] == area, 'Value'].iloc[0] * x[area] for area in areas))
m.setObjective(objective, gp.GRB.MAXIMIZE)
capacity_constraint = gp.quicksum((products_df.loc[products_df['ProductName'] == area, 'Weight'].iloc[0] * x[area] for area in areas))
m.addConstr(capacity_constraint <= total_capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Development Plan ---')
    for area in areas:
        val = x[area].X
        if val > 1e-06:
            print(f'  {area}: {val:.4f} units per day')
else:
    print(f'No optimal solution found. Status: {m.status}')