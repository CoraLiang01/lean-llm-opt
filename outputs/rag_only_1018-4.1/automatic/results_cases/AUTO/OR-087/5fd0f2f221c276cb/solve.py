import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(file_36_1, dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(file_36_2, dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(file_36_3, dtype=str, keep_default_na=False)
products = [f'A{i}' for i in range(1, 81)]

def norm(s):
    return s.strip().casefold()
row_map_36_1 = {norm(row['Product']): idx for idx, row in df_36_1.iterrows()}
row_demand = None
row_price = None
row_cost = None
row_quota = None
for idx, row in df_36_1.iterrows():
    prod = norm(row['Product'])
    if prod == norm('Maximum Demand (100 kg units)'):
        row_demand = idx
    elif prod == norm('Selling Price ($/100 kg)'):
        row_price = idx
    elif prod == norm('Production Cost ($/100 kg)'):
        row_cost = idx
    elif prod == norm('Production Quota (max per day)'):
        row_quota = idx
if None in [row_demand, row_price, row_cost, row_quota]:
    raise ValueError('One or more required rows missing in 36-1.csv')
max_demand = {}
selling_price = {}
production_cost = {}
prod_quota_per_day = {}
for p in products:
    try:
        max_demand[p] = int(float(df_36_1.loc[row_demand, p]))
        selling_price[p] = float(df_36_1.loc[row_price, p])
        production_cost[p] = float(df_36_1.loc[row_cost, p])
        prod_quota_per_day[p] = int(float(df_36_1.loc[row_quota, p]))
    except Exception as e:
        raise ValueError(f'Missing or malformed data for product {p} in 36-1.csv: {e}')
if df_36_2.shape[0] != 1:
    raise ValueError('36-2.csv should have exactly one row for activation costs')
activation_cost = {}
row = df_36_2.iloc[0]
for p in products:
    try:
        activation_cost[p] = float(row[p])
    except Exception as e:
        raise ValueError(f'Missing or malformed activation cost for product {p} in 36-2.csv: {e}')
if df_36_3.shape[0] != 1:
    raise ValueError('36-3.csv should have exactly one row for minimum batch sizes')
min_batch_size = {}
row = df_36_3.iloc[0]
for p in products:
    try:
        min_batch_size[p] = int(float(row[p]))
    except Exception as e:
        raise ValueError(f'Missing or malformed minimum batch size for product {p} in 36-3.csv: {e}')
num_days = 22
m = Model('ProductionPlanning')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(products, vtype=GRB.BINARY, name='')
for p in products:
    m.addConstr(x_vars[p] <= max_demand[p], name=f'demand_{p}')
for p in products:
    m.addConstr(x_vars[p] <= prod_quota_per_day[p] * num_days, name=f'quota_{p}')
for p in products:
    m.addConstr(x_vars[p] >= min_batch_size[p] * y_vars[p], name=f'minbatch_{p}')
for p in products:
    m.addConstr(x_vars[p] <= max_demand[p] * y_vars[p], name=f'link_{p}')
profit_terms = []
for p in products:
    profit_terms.append((selling_price[p] - production_cost[p]) * x_vars[p] - activation_cost[p] * y_vars[p])
m.setObjective(quicksum(profit_terms), GRB.MAXIMIZE)
m.optimize()