import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
if 'product' not in products_df.columns:
    raise KeyError("Missing 'product' column in factory_products_100.csv")
product_ids = products_df['product'].tolist()
if len(product_ids) != 100:
    raise ValueError(f'Expected 100 products, got {len(product_ids)}')

def to_float_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing '{col}' column in factory_products_100.csv")
    try:
        return df[col].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to float: {e}")

def to_int_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing '{col}' column in factory_products_100.csv")
    try:
        return df[col].astype(int).to_dict()
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to int: {e}")
profit_per_unit = to_float_col(products_df, 'profit_per_unit')
r1_per_unit = to_float_col(products_df, 'r1_per_unit')
r2_per_unit = to_float_col(products_df, 'r2_per_unit')
r3_per_unit = to_float_col(products_df, 'r3_per_unit')
upper_demand_units = to_int_col(products_df, 'upper_demand_units')
if 'batch_size_units' not in products_df.columns:
    raise KeyError("Missing 'batch_size_units' column in factory_products_100.csv")
batch_size_units_set = set(products_df['batch_size_units'].astype(int).tolist())
if len(batch_size_units_set) != 1 or 10 not in batch_size_units_set:
    raise ValueError(f'batch_size_units must be fixed at 10 for all products, got {batch_size_units_set}')
batch_size_units = 10
if 'resource' not in resources_df.columns or 'capacity' not in resources_df.columns:
    raise KeyError("Missing 'resource' or 'capacity' column in resources_capacities.csv")
resource_ids = resources_df['resource'].tolist()
if set(resource_ids) != {'R1', 'R2', 'R3'}:
    raise ValueError(f'Expected resources R1, R2, R3, got {resource_ids}')
capacity = {}
for (idx, row) in resources_df.iterrows():
    rid = row['resource']
    try:
        capacity[rid] = float(row['capacity'])
    except Exception as e:
        raise ValueError(f"Failed to convert capacity for resource '{rid}': {e}")
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
for r in ['R1', 'R2', 'R3']:
    if r == 'R1':
        r_per_unit = r1_per_unit
    elif r == 'R2':
        r_per_unit = r2_per_unit
    elif r == 'R3':
        r_per_unit = r3_per_unit
    else:
        raise ValueError(f'Unexpected resource: {r}')
    m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r_per_unit[i] for i in product_ids)) <= capacity[r], name=f'resource_{r}')
for i in product_ids:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()