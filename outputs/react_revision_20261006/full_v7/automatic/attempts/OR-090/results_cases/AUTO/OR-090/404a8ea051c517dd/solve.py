import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
product_keys = products_df['product'].astype(str).tolist()
if len(set(product_keys)) != 100:
    raise ValueError('Expected 100 unique products, got %d' % len(set(product_keys)))

def col_float(df, col):
    return pd.to_numeric(df[col], errors='raise')
profit_per_unit = dict(zip(product_keys, col_float(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(product_keys, col_float(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(product_keys, col_float(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(product_keys, col_float(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(product_keys, pd.to_numeric(products_df['upper_demand_units'], errors='raise', downcast='integer')))
batch_size_units_set = set(pd.to_numeric(products_df['batch_size_units'], errors='raise'))
if len(batch_size_units_set) != 1:
    raise ValueError('batch_size_units is not constant across products')
batch_size_units = batch_size_units_set.pop()
if batch_size_units != 10:
    raise ValueError('batch_size_units is not 10 as expected')
resource_keys = resources_df['resource'].str.strip().str.upper().tolist()
capacity = dict(zip(resource_keys, pd.to_numeric(resources_df['capacity'], errors='raise')))
for rk in ['R1', 'R2', 'R3']:
    if rk not in capacity:
        raise ValueError(f'Resource {rk} not found in resources_capacities.csv')
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_keys)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r1_per_unit[i] for i in product_keys)) <= capacity['R1'], name='r1_cap')
m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r2_per_unit[i] for i in product_keys)) <= capacity['R2'], name='r2_cap')
m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r3_per_unit[i] for i in product_keys)) <= capacity['R3'], name='r3_cap')
for i in product_keys:
    m.addConstr(x_vars[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in product_keys:
        print(f'{x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.status}')