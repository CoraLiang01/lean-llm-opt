import gurobipy as gp
import pandas as pd
import numpy as np
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df1 = pd.read_csv(file_36_1, sep=',')
df2 = pd.read_csv(file_36_2, sep=',')
df3 = pd.read_csv(file_36_3, sep=',')
product_cols = [col for col in df1.columns if col.startswith('A')]
if len(product_cols) != 80:
    raise ValueError(f'Expected 80 products (A1-A80), got {len(product_cols)}')
products = product_cols

def get_row_value(df, row_name, col):
    row = df[df['Product'].str.casefold().str.strip() == row_name.casefold().strip()]
    if row.empty:
        raise ValueError(f"Row '{row_name}' not found in file.")
    return row.iloc[0][col]
max_demand = {}
selling_price = {}
prod_cost = {}
prod_quota = {}
for p in products:
    max_demand[p] = float(get_row_value(df1, 'Maximum Demand (100 kg units)', p))
    selling_price[p] = float(get_row_value(df1, 'Selling Price ($/100 kg)', p))
    prod_cost[p] = float(get_row_value(df1, 'Production Cost ($/100 kg)', p))
    prod_quota[p] = float(get_row_value(df1, 'Production Quota (max per day)', p))
activation_cost = {}
row2 = df2[df2['Product'].str.casefold().str.strip() == 'activation cost ($)'.casefold().strip()]
if row2.empty:
    raise ValueError("Row 'Activation Cost ($)' not found in 36-2.csv.")
row2 = row2.iloc[0]
for p in products:
    activation_cost[p] = float(row2[p])
min_batch = {}
row3 = df3[df3['Product'].str.casefold().str.strip() == 'minimum batch size (100 kg units)'.casefold().strip()]
if row3.empty:
    raise ValueError("Row 'Minimum Batch Size (100 kg units)' not found in 36-3.csv.")
row3 = row3.iloc[0]
for p in products:
    min_batch[p] = float(row3[p])
num_days = 22
m = gp.Model('ProductionPlan')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(products, vtype=gp.GRB.BINARY, name='')
expr = gp.quicksum(((selling_price[p] - prod_cost[p]) * x[p] - activation_cost[p] * y[p] for p in products))
m.setObjective(expr, gp.GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= max_demand[p], name=f'demand_{p}')
    m.addConstr(x[p] <= prod_quota[p] * num_days, name=f'capacity_{p}')
    m.addConstr(x[p] >= min_batch[p] * y[p], name=f'minbatch_{p}')
    m.addConstr(x[p] <= prod_quota[p] * num_days * y[p], name=f'link_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for p in products:
        print(f'{x[p].VarName} {x[p].X}')
        print(f'{y[p].VarName} {y[p].X}')
else:
    print(f'Solver status: {m.status}')