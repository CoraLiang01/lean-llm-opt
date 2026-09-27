import gurobipy as gp
import pandas as pd
import numpy as np
import re

def get_row_by_label(df, label):

    def norm(s):
        return re.sub('\\s+', ' ', str(s)).strip().casefold()
    label_norm = norm(label)
    for idx, row in df.iterrows():
        if norm(row['Product']) == label_norm:
            return row
    raise KeyError(f"Row with label '{label}' not found in DataFrame.")
path1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(path1, sep=',')
df2 = pd.read_csv(path2, sep=',')
df3 = pd.read_csv(path3, sep=',')
product_cols = [col for col in df1.columns if col.startswith('A')]
products = product_cols
max_demand_row = get_row_by_label(df1, 'Maximum Demand (100 kg units)')
selling_price_row = get_row_by_label(df1, 'Selling Price ($/100 kg)')
prod_cost_row = get_row_by_label(df1, 'Production Cost ($/100 kg)')
quota_row = get_row_by_label(df1, 'Production Quota (max per day)')
max_demand = {p: float(max_demand_row[p]) for p in products}
selling_price = {p: float(selling_price_row[p]) for p in products}
prod_cost = {p: float(prod_cost_row[p]) for p in products}
quota = {p: float(quota_row[p]) for p in products}
activation_cost_row = df2.iloc[0]
activation_cost = {p: float(activation_cost_row[p]) for p in products}
min_batch_row = df3.iloc[0]
min_batch = {p: float(min_batch_row[p]) for p in products}
for p in products:
    for param, d in [('max_demand', max_demand), ('selling_price', selling_price), ('prod_cost', prod_cost), ('quota', quota), ('activation_cost', activation_cost), ('min_batch', min_batch)]:
        if p not in d:
            raise KeyError(f'Missing {param} for product {p}')
m = gp.Model('ProductionPlan80')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
profit_terms = [(selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products]
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= max_demand[p] for p in products), name='')
m.addConstrs((x[p] <= 22 * quota[p] for p in products), name='')
m.addConstrs((x[p] >= min_batch[p] * y[p] for p in products), name='')
m.addConstrs((x[p] <= max_demand[p] * y[p] for p in products), name='')
m.addConstr(gp.quicksum((x[p] / quota[p] for p in products)) <= 22, name='shared_days')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for p in products:
        if y[p].X > 0.5:
            print(f'Product {p}: ACTIVE (y={int(round(y[p].X))}), Production: {int(round(x[p].X))} (100kg units)')
        else:
            print(f'Product {p}: INACTIVE (y={int(round(y[p].X))}), Production: {int(round(x[p].X))} (100kg units)')
else:
    print(f'No optimal solution found. Status: {m.status}')