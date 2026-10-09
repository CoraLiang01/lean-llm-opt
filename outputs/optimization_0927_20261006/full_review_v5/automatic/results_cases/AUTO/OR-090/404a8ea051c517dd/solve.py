import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv'
resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
resources_df = pd.read_csv(resources_path, dtype=str, keep_default_na=False)
if 'product' not in products_df.columns:
    raise KeyError("Missing required column 'product' in products CSV.")
product_ids = products_df['product'].astype(str).tolist()

def to_float_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in products CSV.")
    return df[col].astype(float).values

def to_int_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in products CSV.")
    return df[col].astype(int).values
profit_per_unit = dict(zip(product_ids, to_float_col(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(product_ids, to_float_col(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(product_ids, to_float_col(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(product_ids, to_float_col(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(product_ids, to_int_col(products_df, 'upper_demand_units')))
if 'batch_size_units' not in products_df.columns:
    raise KeyError("Missing required column 'batch_size_units' in products CSV.")
batch_size_units_set = set(to_int_col(products_df, 'batch_size_units'))
if len(batch_size_units_set) != 1:
    raise ValueError('batch_size_units is not constant across all products.')
batch_size_units = batch_size_units_set.pop()
if 'resource' not in resources_df.columns or 'capacity' not in resources_df.columns:
    raise KeyError('Missing required columns in resources CSV.')
resources_df['resource_norm'] = resources_df['resource'].str.strip().str.casefold()
resource_ids = resources_df['resource'].tolist()
resource_norm_map = dict(zip(resources_df['resource_norm'], resources_df['resource']))
resource_caps = {}
for r in ['r1', 'r2', 'r3']:
    norm = r.strip().casefold()
    if norm not in resource_norm_map:
        raise KeyError(f"Resource '{r}' not found in resources CSV.")
    business_id = resource_norm_map[norm]
    cap_val = float(resources_df.loc[resources_df['resource_norm'] == norm, 'capacity'].values[0])
    resource_caps[r] = cap_val
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r1_per_unit[i] for i in product_ids)) <= resource_caps['r1'], name='resource_r1')
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r2_per_unit[i] for i in product_ids)) <= resource_caps['r2'], name='resource_r2')
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r3_per_unit[i] for i in product_ids)) <= resource_caps['r3'], name='resource_r3')
for i in product_ids:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()