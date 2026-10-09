import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
products_df['product'] = products_df['product'].str.strip()
product_ids = products_df['product'].tolist()

def to_float(series, col):
    return series[col].astype(float)

def to_int(series, col):
    return series[col].astype(int)
profit_per_unit = dict(zip(products_df['product'], to_float(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(products_df['product'], to_float(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(products_df['product'], to_float(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(products_df['product'], to_float(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(products_df['product'], to_int(products_df, 'upper_demand_units')))
batch_size_set = set(to_int(products_df, 'batch_size_units'))
if len(batch_size_set) != 1:
    raise ValueError('batch_size_units is not unique across products.')
batch_size_units = batch_size_set.pop()
resources_df['resource'] = resources_df['resource'].str.strip()
resource_ids = resources_df['resource'].tolist()
capacity = dict(zip(resources_df['resource'], resources_df['capacity'].astype(float)))
expected_resources = ['R1', 'R2', 'R3']
if sorted(resource_ids) != sorted(expected_resources):
    raise ValueError(f'Resource IDs in CSV do not match expected {expected_resources}')
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((batch_size_units * r1_per_unit[i] * x_vars[i] for i in product_ids)) <= capacity['R1'], name='resource_R1')
m.addConstr(gp.quicksum((batch_size_units * r2_per_unit[i] * x_vars[i] for i in product_ids)) <= capacity['R2'], name='resource_R2')
m.addConstr(gp.quicksum((batch_size_units * r3_per_unit[i] * x_vars[i] for i in product_ids)) <= capacity['R3'], name='resource_R3')
for i in product_ids:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan (batches, units, profit) ---')
    for i in product_ids:
        batches = int(round(x_vars[i].X))
        units = batches * batch_size_units
        if batches > 0:
            profit = units * profit_per_unit[i]
            print(f'{i}: {batches} batches ({units} units), profit: {profit:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')