import gurobipy as gp
import pandas as pd
import numpy as np
path_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(path_36_1, dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(path_36_2, dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(path_36_3, dtype=str, keep_default_na=False)
product_ids = [col for col in df_36_1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)}')

def get_row_by_label(df, label):
    norm_label = label.strip().casefold()
    for (idx, val) in enumerate(df['Product']):
        if val.strip().casefold() == norm_label:
            return df.iloc[idx]
    raise KeyError(f"Row '{label}' not found in file.")
row_demand = get_row_by_label(df_36_1, 'Maximum Demand (100 kg units)')
row_price = get_row_by_label(df_36_1, 'Selling Price ($/100 kg)')
row_cost = get_row_by_label(df_36_1, 'Production Cost ($/100 kg)')
row_quota = get_row_by_label(df_36_1, 'Production Quota (max per day)')
row_activation = get_row_by_label(df_36_2, 'Activation Cost ($)')
row_minbatch = get_row_by_label(df_36_3, 'Minimum Batch Size (100 kg units)')

def to_float_dict(row, keys):
    return {k: float(row[k]) for k in keys}

def to_int_dict(row, keys):
    return {k: int(float(row[k])) for k in keys}
max_demand = to_float_dict(row_demand, product_ids)
selling_price = to_float_dict(row_price, product_ids)
production_cost = to_float_dict(row_cost, product_ids)
daily_quota = to_float_dict(row_quota, product_ids)
activation_cost = to_float_dict(row_activation, product_ids)
min_batch = to_int_dict(row_minbatch, product_ids)
m = gp.Model('ProductionPlan80')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = [(selling_price[i] - production_cost[i]) * x_vars[i] - activation_cost[i] * y_vars[i] for i in product_ids]
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= max_demand[i] for i in product_ids), name='')
m.addConstrs((x_vars[i] <= 22 * daily_quota[i] for i in product_ids), name='')
m.addConstrs((x_vars[i] >= min_batch[i] * y_vars[i] for i in product_ids), name='')
m.addConstrs((x_vars[i] <= max_demand[i] * y_vars[i] for i in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[i] / daily_quota[i] for i in product_ids)) <= 22, name='TotalProductionDays')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for i in product_ids:
        x_val = x_vars[i].X
        y_val = y_vars[i].X
        if y_val > 0.5:
            print(f'{i}: Produce {int(round(x_val))} (100kg units), Activated (y=1)')
        else:
            print(f'{i}: Not produced (y=0)')
else:
    print(f'No optimal solution found. Status: {m.status}')