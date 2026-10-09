import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
products_df['profit_per_unit'] = products_df['profit_per_unit'].astype(float)
products_df['r1_per_unit'] = products_df['r1_per_unit'].astype(float)
products_df['r2_per_unit'] = products_df['r2_per_unit'].astype(float)
products_df['r3_per_unit'] = products_df['r3_per_unit'].astype(float)
products_df['upper_demand_units'] = products_df['upper_demand_units'].astype(int)
products_df['batch_size_units'] = products_df['batch_size_units'].astype(int)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
resources_df['capacity'] = resources_df['capacity'].astype(float)
product_ids = products_df['product'].tolist()
resource_ids = resources_df['resource'].tolist()
profit_per_unit = dict(zip(products_df['product'], products_df['profit_per_unit']))
r1_per_unit = dict(zip(products_df['product'], products_df['r1_per_unit']))
r2_per_unit = dict(zip(products_df['product'], products_df['r2_per_unit']))
r3_per_unit = dict(zip(products_df['product'], products_df['r3_per_unit']))
upper_demand_units = dict(zip(products_df['product'], products_df['upper_demand_units']))
batch_size_units_dict = dict(zip(products_df['product'], products_df['batch_size_units']))
batch_sizes = set(batch_size_units_dict.values())
if len(batch_sizes) != 1 or 10 not in batch_sizes:
    raise ValueError('All batch_size_units must be 10 for all products.')
batch_size_units = 10
resource_capacity = dict(zip(resources_df['resource'], resources_df['capacity']))
resource_consumption = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
m = Model('factory_batch_production')
x_vars = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), GRB.MAXIMIZE)
for r in resource_ids:
    m.addConstr(quicksum((batch_size_units * x_vars[i] * resource_consumption[r][i] for i in product_ids)) <= resource_capacity[r], name=f'resource_{r}_capacity')
for i in product_ids:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}_upper')
m.optimize()