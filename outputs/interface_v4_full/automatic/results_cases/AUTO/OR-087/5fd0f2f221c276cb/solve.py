import gurobipy as gp
import pandas as pd
import numpy as np
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
product_cols = [col for col in df1.columns if col != 'Product']
products = product_cols

def get_row(df, row_name):
    idx = df['Product'].str.strip().str.casefold() == row_name.strip().casefold()
    if not idx.any():
        raise KeyError(f"Row '{row_name}' not found in file.")
    return df.loc[idx].iloc[0]
max_demand_row = get_row(df1, 'Maximum Demand (100 kg units)')
selling_price_row = get_row(df1, 'Selling Price ($/100 kg)')
prod_cost_row = get_row(df1, 'Production Cost ($/100 kg)')
quota_row = get_row(df1, 'Production Quota (max per day)')
max_demand = {p: float(max_demand_row[p]) for p in products}
selling_price = {p: float(selling_price_row[p]) for p in products}
prod_cost = {p: float(prod_cost_row[p]) for p in products}
quota = {p: float(quota_row[p]) for p in products}
if df2.shape[0] != 1:
    raise ValueError('36-2.csv should have exactly one row with activation costs.')
activation_cost_row = df2.iloc[0]
activation_cost = {p: float(activation_cost_row[p]) for p in products}
if df3.shape[0] != 1:
    raise ValueError('36-3.csv should have exactly one row with minimum batch sizes.')
min_batch_row = df3.iloc[0]
min_batch = {p: float(min_batch_row[p]) for p in products}
num_days = 22
m = gp.Model('ProductionPlan80Products')
x = m.addVars(products, vtype=gp.GRB.INTEGER, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
profit_terms = gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products))
m.setObjective(profit_terms, gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
m.addConstrs((x[p] <= quota[p] * num_days for p in products), name='')
m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
m.addConstr(gp.quicksum((x[p] / quota[p] for p in products)) <= num_days, name='SharedProductionDays')
m.optimize()