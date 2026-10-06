import gurobipy as gp
import pandas as pd
import numpy as np
path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(path_36_1, sep=',')
df_36_2 = pd.read_csv(path_36_2, sep=',')
df_36_3 = pd.read_csv(path_36_3, sep=',')
product_cols = [col for col in df_36_1.columns if col.startswith('A')]
if len(product_cols) != 80:
    raise ValueError(f'Expected 80 products (A1-A80), found {len(product_cols)}: {product_cols}')
products = product_cols

def get_row(df, row_name):
    row = df[df['Product'].str.strip().casefold() == row_name.strip().casefold()]
    if row.empty:
        raise KeyError(f"Row '{row_name}' not found in file {df}")
    return row.iloc[0]
row_demand = get_row(df_36_1, 'Maximum Demand (100 kg units)')
row_price = get_row(df_36_1, 'Selling Price ($/100 kg)')
row_cost = get_row(df_36_1, 'Production Cost ($/100 kg)')
row_quota = get_row(df_36_1, 'Production Quota (max per day)')
row_activation = get_row(df_36_2, 'Activation Cost ($)')
row_minbatch = get_row(df_36_3, 'Minimum Batch Size (100 kg units)')
max_demand = {p: float(row_demand[p]) for p in products}
selling_price = {p: float(row_price[p]) for p in products}
prod_cost = {p: float(row_cost[p]) for p in products}
prod_quota = {p: float(row_quota[p]) for p in products}
activation_cost = {p: float(row_activation[p]) for p in products}
min_batch = {p: float(row_minbatch[p]) for p in products}
m = gp.Model('MonthlyProductionPlan')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
profit_terms = gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products))
m.setObjective(profit_terms, gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
m.addConstrs((x[p] <= 22 * prod_quota[p] for p in products), name='')
m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
m.addConstr(gp.quicksum((x[p] / prod_quota[p] for p in products)) <= 22, name='SharedProdDays')
m.optimize()