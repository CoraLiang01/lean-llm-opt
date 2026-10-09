import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def find_col(df, pattern):
    for col in df.columns:
        if re.fullmatch(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
products_df['product'] = products_df['product'].str.strip()
product_ids = list(products_df['product'])

def to_float_col(df, col):
    return df[col].astype(float)

def to_int_col(df, col):
    return df[col].astype(int)
profit_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(products_df['product'], to_int_col(products_df, 'upper_demand_units')))
batch_size_units_col = to_int_col(products_df, 'batch_size_units')
if not (batch_size_units_col == batch_size_units_col.iloc[0]).all():
    raise ValueError('batch_size_units is not constant across all products.')
batch_size_units = int(batch_size_units_col.iloc[0])
resources_df['resource'] = resources_df['resource'].str.strip().str.casefold()
resource_ids = ['r1', 'r2', 'r3']
resource_capacities = {}
for r in resource_ids:
    cap_row = resources_df[resources_df['resource'] == r]
    if cap_row.empty:
        raise KeyError(f"Resource '{r.upper()}' not found in resources_capacities.csv")
    resource_capacities[r] = float(cap_row['capacity'].iloc[0])
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[i] * batch_size_units * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
for (r, per_unit_dict) in zip(resource_ids, [r1_per_unit, r2_per_unit, r3_per_unit]):
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * per_unit_dict[i] for i in product_ids)) <= resource_capacities[r], name=f'res_{r}')
for i in product_ids:
    m.addConstr(x_vars[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()