import pandas as pd
import numpy as np
from gurobipy import Model, GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products_df['product'] = products_df['product'].astype(str).str.strip()
resources_df['resource'] = resources_df['resource'].astype(str).str.strip()
product_ids = products_df['product'].tolist()
resource_ids = resources_df['resource'].tolist()
batch_size_set = set(products_df['batch_size_units'])
if len(batch_size_set) != 1 or list(batch_size_set)[0] != 10:
    raise ValueError('batch_size_units must be constant and equal to 10 for all products.')
batch_size_units = 10
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
capacity = resources_df.set_index('resource')['capacity'].to_dict()
for pid in product_ids:
    if pid not in profit_per_unit or pid not in upper_demand_units or pid not in r1_per_unit or (pid not in r2_per_unit) or (pid not in r3_per_unit):
        raise ValueError(f'Missing parameter data for product {pid}')
for rid in ['R1', 'R2', 'R3']:
    if rid not in capacity:
        raise ValueError(f'Missing capacity for resource {rid}')
m = Model('factory_batch_production')
x = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(sum((x[pid] * batch_size_units * profit_per_unit[pid] for pid in product_ids)), GRB.MAXIMIZE)
m.addConstr(sum((x[pid] * batch_size_units * r1_per_unit[pid] for pid in product_ids)) <= capacity['R1'], name='res_R1')
m.addConstr(sum((x[pid] * batch_size_units * r2_per_unit[pid] for pid in product_ids)) <= capacity['R2'], name='res_R2')
m.addConstr(sum((x[pid] * batch_size_units * r3_per_unit[pid] for pid in product_ids)) <= capacity['R3'], name='res_R3')
for pid in product_ids:
    m.addConstr(x[pid] * batch_size_units <= upper_demand_units[pid], name=f'demand_{pid}')
m.optimize()