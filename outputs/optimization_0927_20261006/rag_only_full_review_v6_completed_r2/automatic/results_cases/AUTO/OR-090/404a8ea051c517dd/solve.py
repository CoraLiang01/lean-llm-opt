import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
for col in ['profit_per_unit', 'r1_per_unit', 'r2_per_unit', 'r3_per_unit']:
    products_df[col] = products_df[col].astype(float)
products_df['upper_demand_units'] = products_df['upper_demand_units'].astype(int)
products_df['batch_size_units'] = products_df['batch_size_units'].astype(int)
products_df['product'] = products_df['product'].str.strip()
products_df.set_index('product', inplace=True)
product_ids = list(products_df.index)
batch_sizes = products_df['batch_size_units'].unique()
if len(batch_sizes) != 1 or batch_sizes[0] != 10:
    raise ValueError('batch_size_units must be constant and equal to 10 for all products.')
batch_size_units = int(batch_sizes[0])
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
resources_df['resource'] = resources_df['resource'].str.strip()
resources_df['capacity'] = resources_df['capacity'].astype(float)
resource_ids = ['R1', 'R2', 'R3']
missing_resources = set(resource_ids) - set(resources_df['resource'])
if missing_resources:
    raise ValueError(f'Missing resource capacities for: {missing_resources}')
resource_capacity = resources_df.set_index('resource')['capacity'].to_dict()
r1_per_unit = products_df['r1_per_unit'].to_dict()
r2_per_unit = products_df['r2_per_unit'].to_dict()
r3_per_unit = products_df['r3_per_unit'].to_dict()
profit_per_unit = products_df['profit_per_unit'].to_dict()
upper_demand_units = products_df['upper_demand_units'].to_dict()
m = Model('factory_batch_production')
x_vars = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), GRB.MAXIMIZE)
for (r, per_unit_dict) in zip(resource_ids, [r1_per_unit, r2_per_unit, r3_per_unit]):
    m.addConstr(quicksum((batch_size_units * x_vars[i] * per_unit_dict[i] for i in product_ids)) <= resource_capacity[r], name=f'res_{r}')
for i in product_ids:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()