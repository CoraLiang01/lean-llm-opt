import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
product_ids = products_df['product'].tolist()
if len(product_ids) != 100:
    raise ValueError(f'Expected 100 products, found {len(product_ids)}.')

def to_float_col(df, col):
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")

def to_int_col(df, col):
    try:
        return df[col].astype(int)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to int: {e}")
profit_per_unit = dict(zip(product_ids, to_float_col(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(product_ids, to_float_col(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(product_ids, to_float_col(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(product_ids, to_float_col(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(product_ids, to_int_col(products_df, 'upper_demand_units')))
batch_size_units_set = set(to_int_col(products_df, 'batch_size_units'))
if len(batch_size_units_set) != 1:
    raise ValueError(f'Expected a single batch_size_units value, found: {batch_size_units_set}')
batch_size_units = batch_size_units_set.pop()
if batch_size_units != 10:
    raise ValueError(f'Expected batch_size_units=10, got {batch_size_units}')
resource_ids = resources_df['resource'].tolist()
expected_resources = ['R1', 'R2', 'R3']
if set(resource_ids) != set(expected_resources):
    raise ValueError(f'Resource list mismatch. Expected {expected_resources}, got {resource_ids}')
capacity = {}
for (idx, row) in resources_df.iterrows():
    rid = row['resource']
    try:
        capacity[rid] = float(row['capacity'])
    except Exception as e:
        raise ValueError(f"Capacity for resource '{rid}' could not be converted to float: {e}")
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
resource_per_unit = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
for r in expected_resources:
    m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * resource_per_unit[r][i] for i in product_ids)) <= capacity[r], name=f'res_{r}')
for i in product_ids:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()