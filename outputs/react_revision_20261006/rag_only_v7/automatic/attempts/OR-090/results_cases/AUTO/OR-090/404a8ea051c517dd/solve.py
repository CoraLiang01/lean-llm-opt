import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
products_df['product'] = products_df['product'].str.strip()
product_ids = products_df['product'].tolist()
if len(set(product_ids)) != 100:
    raise ValueError('Expected 100 unique products, got %d' % len(set(product_ids)))
for col in ['profit_per_unit', 'r1_per_unit', 'r2_per_unit', 'r3_per_unit']:
    products_df[col] = products_df[col].astype(float)
products_df['upper_demand_units'] = products_df['upper_demand_units'].astype(int)
products_df['batch_size_units'] = products_df['batch_size_units'].astype(int)
batch_sizes = products_df['batch_size_units'].unique()
if len(batch_sizes) != 1 or batch_sizes[0] != 10:
    raise ValueError('Batch size must be unique and equal to 10')
batch_size_units = int(batch_sizes[0])
profit_per_unit = dict(zip(products_df['product'], products_df['profit_per_unit']))
r1_per_unit = dict(zip(products_df['product'], products_df['r1_per_unit']))
r2_per_unit = dict(zip(products_df['product'], products_df['r2_per_unit']))
r3_per_unit = dict(zip(products_df['product'], products_df['r3_per_unit']))
upper_demand_units = dict(zip(products_df['product'], products_df['upper_demand_units']))
resources_df['resource'] = resources_df['resource'].str.strip()
resource_ids = resources_df['resource'].tolist()
expected_resources = ['R1', 'R2', 'R3']
if sorted(resource_ids) != sorted(expected_resources):
    raise ValueError('Resource IDs do not match expected: %s' % expected_resources)
resources_df['capacity'] = resources_df['capacity'].astype(float)
capacity = dict(zip(resources_df['resource'], resources_df['capacity']))
for pid in product_ids:
    for d in [profit_per_unit, r1_per_unit, r2_per_unit, r3_per_unit, upper_demand_units]:
        if pid not in d:
            raise ValueError(f'Missing parameter for product {pid}')

def solve_problem():
    m = gp.Model('factory_batch_production')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(product_ids, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((x_vars[pid] * batch_size_units * profit_per_unit[pid] for pid in product_ids)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x_vars[pid] * batch_size_units * r1_per_unit[pid] for pid in product_ids)) <= capacity['R1'], name='res_R1')
    m.addConstr(gp.quicksum((x_vars[pid] * batch_size_units * r2_per_unit[pid] for pid in product_ids)) <= capacity['R2'], name='res_R2')
    m.addConstr(gp.quicksum((x_vars[pid] * batch_size_units * r3_per_unit[pid] for pid in product_ids)) <= capacity['R3'], name='res_R3')
    for pid in product_ids:
        m.addConstr(x_vars[pid] * batch_size_units <= upper_demand_units[pid], name=f'demand_{pid}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')