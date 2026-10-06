import pandas as pd
import numpy as np
from gurobipy import Model, GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products_df['product'] = products_df['product'].astype(str).str.strip()
resources_df['resource'] = resources_df['resource'].astype(str).str.strip()
product_ids = list(products_df['product'])
resource_ids = list(resources_df['resource'])
batch_size_set = set(products_df['batch_size_units'])
if len(batch_size_set) != 1 or list(batch_size_set)[0] != 10:
    raise ValueError('batch_size_units must be 10 for all products.')
batch_size_units = 10
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
resource_capacity = resources_df.set_index('resource')['capacity'].to_dict()
resource_consumption = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
if set(resource_ids) != {'R1', 'R2', 'R3'}:
    raise ValueError('Resource IDs must be exactly R1, R2, R3.')
if len(product_ids) != 100:
    raise ValueError('There must be exactly 100 products.')
m = Model('factory_batch_production')
x = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(sum((batch_size_units * x[i] * profit_per_unit[i] for i in product_ids)), GRB.MAXIMIZE)
for r in resource_ids:
    m.addConstr(sum((batch_size_units * x[i] * resource_consumption[r][i] for i in product_ids)) <= resource_capacity[r], name=f'res_{r}')
for i in product_ids:
    m.addConstr(batch_size_units * x[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()