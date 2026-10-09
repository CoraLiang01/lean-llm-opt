import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', dtype=str, keep_default_na=False)
product_ids = products_df['product'].astype(str).tolist()
batch_size_col = 'batch_size_units'
if batch_size_col not in products_df.columns:
    raise KeyError(f"Missing required column '{batch_size_col}' in products file.")
batch_sizes = products_df[batch_size_col].astype(float).unique()
if len(batch_sizes) != 1:
    raise ValueError('Batch size must be the same for all products.')
batch_size_units = int(batch_sizes[0])
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
resource_usage_per_unit = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
resource_ids = resources_df['resource'].astype(str).tolist()
capacity = resources_df.set_index('resource')['capacity'].astype(float).to_dict()
expected_resources = {'R1', 'R2', 'R3'}
if set(resource_ids) != expected_resources:
    raise ValueError(f'Resource set mismatch. Expected {expected_resources}, got {set(resource_ids)}.')
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[pid] * profit_per_unit[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
for r in expected_resources:
    m.addConstr(gp.quicksum((batch_size_units * x_vars[pid] * resource_usage_per_unit[r][pid] for pid in product_ids)) <= capacity[r], name=f'resource_{r}_capacity')
for pid in product_ids:
    m.addConstr(batch_size_units * x_vars[pid] <= upper_demand_units[pid], name=f'upper_demand_{pid}')
m.optimize()