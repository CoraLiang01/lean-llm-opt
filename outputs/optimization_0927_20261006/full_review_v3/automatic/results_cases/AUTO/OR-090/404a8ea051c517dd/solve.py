import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
products_df['product'] = products_df['product'].str.strip()
product_ids = products_df['product'].tolist()
products_df['profit_per_unit'] = products_df['profit_per_unit'].astype(float)
products_df['r1_per_unit'] = products_df['r1_per_unit'].astype(float)
products_df['r2_per_unit'] = products_df['r2_per_unit'].astype(float)
products_df['r3_per_unit'] = products_df['r3_per_unit'].astype(float)
products_df['upper_demand_units'] = products_df['upper_demand_units'].astype(int)
products_df['batch_size_units'] = products_df['batch_size_units'].astype(int)
batch_size_set = set(products_df['batch_size_units'].unique())
if len(batch_size_set) != 1 or 10 not in batch_size_set:
    raise ValueError(f'Batch size is not constant 10: found {batch_size_set}')
batch_size_units = 10
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
resources_df['resource'] = resources_df['resource'].str.strip()
resources_df['capacity'] = resources_df['capacity'].astype(float)
resource_ids = ['R1', 'R2', 'R3']
resource_capacities = {}
for r in resource_ids:
    cap_row = resources_df[resources_df['resource'].str.casefold() == r.casefold()]
    if cap_row.empty:
        raise ValueError(f'Resource {r} not found in resources_capacities.csv')
    resource_capacities[r] = float(cap_row['capacity'].iloc[0])
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
resource_coeffs = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
for r in resource_ids:
    m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * resource_coeffs[r][i] for i in product_ids)) <= resource_capacities[r], name=f'res_{r}')
for i in product_ids:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()