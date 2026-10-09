import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
if 'product' not in products_df.columns:
    raise KeyError("Missing required column 'product' in products CSV.")
products_df['product'] = products_df['product'].str.strip()
products = products_df['product'].tolist()

def to_float_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in products CSV.")
    return df[col].astype(float)

def to_int_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in products CSV.")
    return df[col].astype(int)
profit_per_unit = dict(zip(products, to_float_col(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(products, to_float_col(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(products, to_float_col(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(products, to_float_col(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(products, to_int_col(products_df, 'upper_demand_units')))
batch_size_units_col = 'batch_size_units'
if batch_size_units_col not in products_df.columns:
    raise KeyError(f"Missing required column '{batch_size_units_col}' in products CSV.")
batch_size_units_set = set(to_int_col(products_df, batch_size_units_col))
if len(batch_size_units_set) != 1:
    raise ValueError('All products must have the same batch_size_units as per query; found: %s' % batch_size_units_set)
batch_size_units = batch_size_units_set.pop()
if 'resource' not in resources_df.columns or 'capacity' not in resources_df.columns:
    raise KeyError('Missing required columns in resources_capacities CSV.')
resources_df['resource'] = resources_df['resource'].str.strip()
resources = resources_df['resource'].tolist()
capacity = dict(zip(resources, resources_df['capacity'].astype(float)))
for r in ['R1', 'R2', 'R3']:
    if r not in capacity:
        raise KeyError(f"Resource '{r}' not found in resources_capacities.csv.")
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in products)), gp.GRB.MAXIMIZE)
resource_coeffs = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
for r in ['R1', 'R2', 'R3']:
    m.addConstr(gp.quicksum((batch_size_units * resource_coeffs[r][i] * x_vars[i] for i in products)) <= capacity[r], name=f'res_{r}')
for i in products:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()