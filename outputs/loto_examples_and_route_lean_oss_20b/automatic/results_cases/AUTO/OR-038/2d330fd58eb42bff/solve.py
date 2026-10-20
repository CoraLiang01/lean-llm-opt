import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
cap_dict = dict(zip(capacity_df['VehicleType'], capacity_df['Capacity']))
benefit_dict = dict(zip(products_df['ProductName'], products_df['Value']))
for v in cap_dict:
    if v not in benefit_dict:
        raise ValueError(f"Benefit for vehicle type '{v}' not found in products.csv")
V = list(cap_dict.keys())
total_cap = sum((cap_dict[v] for v in V))
m = gp.Model('NewCarSalesInventory')
x = m.addVars(V, vtype=gp.GRB.INTEGER)
for v in V:
    m.addConstr(x[v] <= cap_dict[v])
m.addConstr(gp.quicksum((x[v] for v in V)) <= total_cap)
m.setObjective(gp.quicksum((benefit_dict[v] * x[v] for v in V)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Ordering Plan ---')
    for v in V:
        print(f'{v}: {x[v].X:.0f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')