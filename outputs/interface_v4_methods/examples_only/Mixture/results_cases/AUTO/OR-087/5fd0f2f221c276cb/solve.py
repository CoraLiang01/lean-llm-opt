import gurobipy as gp
import pandas as pd
import numpy as np
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(file_36_1, sep=',')
df2 = pd.read_csv(file_36_2, sep=',')
df3 = pd.read_csv(file_36_3, sep=',')
product_cols = [f'A{i}' for i in range(1, 81)]
products = product_cols.copy()
row_max_demand = df1[df1['Product'].str.strip().casefold() == 'maximum demand (100 kg units)'.casefold()]
row_selling_price = df1[df1['Product'].str.strip().casefold() == 'selling price ($/100 kg)'.casefold()]
row_prod_cost = df1[df1['Product'].str.strip().casefold() == 'production cost ($/100 kg)'.casefold()]
row_daily_quota = df1[df1['Product'].str.strip().casefold() == 'production quota (max per day)'.casefold()]
if row_max_demand.empty or row_selling_price.empty or row_prod_cost.empty or row_daily_quota.empty:
    raise ValueError('One or more required rows are missing in 36-1.csv.')
max_demand = {p: float(row_max_demand.iloc[0][p]) for p in products}
selling_price = {p: float(row_selling_price.iloc[0][p]) for p in products}
prod_cost = {p: float(row_prod_cost.iloc[0][p]) for p in products}
daily_quota = {p: float(row_daily_quota.iloc[0][p]) for p in products}
row_activation_cost = df2[df2['Product'].str.strip().casefold() == 'activation cost ($)'.casefold()]
if row_activation_cost.empty:
    raise ValueError('Activation Cost row missing in 36-2.csv.')
activation_cost = {p: float(row_activation_cost.iloc[0][p]) for p in products}
row_min_batch = df3[df3['Product'].str.strip().casefold() == 'minimum batch size (100 kg units)'.casefold()]
if row_min_batch.empty:
    raise ValueError('Minimum Batch Size row missing in 36-3.csv.')
min_batch = {p: int(row_min_batch.iloc[0][p]) for p in products}
num_days = 22
m = gp.Model('ProductionPlanFixedCharge')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
profit_terms = [(selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products]
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= max_demand[p])
    m.addConstr(x[p] <= daily_quota[p] * num_days)
    m.addConstr(x[p] >= min_batch[p] * y[p])
    m.addConstr(x[p] <= min(max_demand[p], daily_quota[p] * num_days) * y[p])
m.optimize()