import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
df_36_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv', sep=',', dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv', sep=',', dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv', sep=',', dtype=str, keep_default_na=False)
product_id_col = None
for col in df_36_1.columns:
    if re.search('product', col, re.IGNORECASE):
        product_id_col = col
        break
if product_id_col is None:
    raise KeyError('Could not find a product identifier column in 36-1.csv')
product_ids = df_36_1[product_id_col].tolist()
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)} in 36-1.csv')

def get_numeric_series(df, id_col, value_col, ids, float_type=float):
    s = df.set_index(id_col)[value_col]
    s.index = s.index.str.strip()
    d = {}
    for pid in ids:
        key = str(pid).strip()
        if key not in s.index:
            raise KeyError(f"Product ID '{key}' not found in column '{id_col}'")
        val = s.loc[key]
        try:
            d[pid] = float_type(val)
        except Exception:
            raise ValueError(f"Could not convert value '{val}' for product '{pid}' in column '{value_col}' to {float_type}")
    return d
selling_price_col = find_col(df_36_1, 'selling\\s*price')
production_cost_col = find_col(df_36_1, 'production\\s*cost')
max_demand_col = find_col(df_36_1, 'maximum\\s*demand')
prod_quota_col = find_col(df_36_1, 'production\\s*quota')
selling_price = get_numeric_series(df_36_1, product_id_col, selling_price_col, product_ids)
production_cost = get_numeric_series(df_36_1, product_id_col, production_cost_col, product_ids)
max_demand = get_numeric_series(df_36_1, product_id_col, max_demand_col, product_ids, int)
prod_quota = get_numeric_series(df_36_1, product_id_col, prod_quota_col, product_ids, float)
product_id_col_2 = None
for col in df_36_2.columns:
    if re.search('product', col, re.IGNORECASE):
        product_id_col_2 = col
        break
if product_id_col_2 is None:
    raise KeyError('Could not find a product identifier column in 36-2.csv')
activation_cost_col = find_col(df_36_2, 'activation\\s*cost')
activation_cost = get_numeric_series(df_36_2, product_id_col_2, activation_cost_col, product_ids)
product_id_col_3 = None
for col in df_36_3.columns:
    if re.search('product', col, re.IGNORECASE):
        product_id_col_3 = col
        break
if product_id_col_3 is None:
    raise KeyError('Could not find a product identifier column in 36-3.csv')
min_batch_col = find_col(df_36_3, 'minimum\\s*batch')
min_batch_size = get_numeric_series(df_36_3, product_id_col_3, min_batch_col, product_ids, int)
m = gp.Model('ProductionPlan80Products')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = ((selling_price[pid] - production_cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids)
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
for pid in product_ids:
    m.addConstr(x_vars[pid] <= max_demand[pid], name=f'demand_{pid}')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= 22 * prod_quota[pid], name=f'quota_{pid}')
for pid in product_ids:
    m.addConstr(x_vars[pid] >= min_batch_size[pid] * y_vars[pid], name=f'minbatch_{pid}')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= max_demand[pid] * y_vars[pid], name=f'link_{pid}')
m.addConstr(gp.quicksum((x_vars[pid] / prod_quota[pid] for pid in product_ids)) <= 22, name='total_days')
m.optimize()