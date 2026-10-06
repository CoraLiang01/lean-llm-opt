import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
product_cols = [f'A{i}' for i in range(1, 81)]
num_days = 22

def get_row(df, row_name):
    matches = df['Product'].apply(lambda x: str(x).casefold().strip() == row_name.casefold().strip())
    if not matches.any():
        raise ValueError(f"Row '{row_name}' not found in DataFrame")
    return df.loc[matches, product_cols].iloc[0]
max_demand = get_row(df1, 'Maximum Demand (100 kg units)').astype(float).to_dict()
selling_price = get_row(df1, 'Selling Price ($/100 kg)').astype(float).to_dict()
prod_cost = get_row(df1, 'Production Cost ($/100 kg)').astype(float).to_dict()
prod_quota = get_row(df1, 'Production Quota (max per day)').astype(float).to_dict()
activation_cost = df2.loc[0, product_cols].astype(float).to_dict()
min_batch_size = df3.loc[0, product_cols].astype(float).to_dict()
for param_name, param in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch_size', min_batch_size)]:
    missing = set(product_cols) - set(param.keys())
    if missing:
        raise ValueError(f'Missing products in {param_name}: {missing}')
m = gp.Model('ProductionPlanning')
x = m.addVars(product_cols, vtype=GRB.INTEGER, name='')
y = m.addVars(product_cols, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in product_cols)), GRB.MAXIMIZE)
for p in product_cols:
    m.addConstr(x[p] <= max_demand[p], name=f'demand_{p}')
for p in product_cols:
    m.addConstr(x[p] <= prod_quota[p] * num_days, name=f'capacity_{p}')
for p in product_cols:
    m.addConstr(x[p] >= min_batch_size[p] * y[p], name=f'minbatch_{p}')
for p in product_cols:
    m.addConstr(x[p] <= prod_quota[p] * num_days * y[p], name=f'link_{p}')
m.optimize()