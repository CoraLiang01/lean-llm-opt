import pandas as pd
import numpy as np
from gurobipy import Model, GRB
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(file_36_1, sep=',')
df_36_2 = pd.read_csv(file_36_2, sep=',')
df_36_3 = pd.read_csv(file_36_3, sep=',')
product_cols = [f'A{i}' for i in range(1, 81)]
num_products = 80
num_days = 22

def get_row_by_label(df, label):
    df_labels = df['Product'].str.strip().str.casefold()
    idx = df_labels[df_labels == label.strip().casefold()].index
    if len(idx) == 0:
        raise ValueError(f"Row '{label}' not found in 36-1.csv")
    return df.loc[idx[0], product_cols]
max_demand = get_row_by_label(df_36_1, 'Maximum Demand (100 kg units)').astype(float).to_dict()
selling_price = get_row_by_label(df_36_1, 'Selling Price ($/100 kg)').astype(float).to_dict()
prod_cost = get_row_by_label(df_36_1, 'Production Cost ($/100 kg)').astype(float).to_dict()
prod_quota = get_row_by_label(df_36_1, 'Production Quota (max per day)').astype(float).to_dict()
activation_cost_row = df_36_2.iloc[0]
activation_cost = activation_cost_row[product_cols].astype(float).to_dict()
min_batch_row = df_36_3.iloc[0]
min_batch_size = min_batch_row[product_cols].astype(float).to_dict()
for (param_name, param_dict) in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('prod_quota', prod_quota), ('activation_cost', activation_cost), ('min_batch_size', min_batch_size)]:
    if set(param_dict.keys()) != set(product_cols):
        raise ValueError(f"Parameter '{param_name}' does not cover all products.")
m = Model('ProductionPlanning')
x = m.addVars(product_cols, vtype=GRB.INTEGER, name='')
y = m.addVars(product_cols, vtype=GRB.BINARY, name='')
m.setObjective(sum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in product_cols)), GRB.MAXIMIZE)
for p in product_cols:
    m.addConstr(x[p] <= max_demand[p], name=f'demand_{p}')
for p in product_cols:
    m.addConstr(x[p] <= prod_quota[p] * num_days, name=f'capacity_{p}')
for p in product_cols:
    m.addConstr(x[p] >= min_batch_size[p] * y[p], name=f'minbatch_{p}')
for p in product_cols:
    m.addConstr(x[p] <= prod_quota[p] * num_days * y[p], name=f'link_{p}')
m.optimize()