import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
products_df['product'] = products_df['product'].str.strip()
resources_df['resource'] = resources_df['resource'].str.strip()
product_ids = products_df['product'].tolist()
if len(set(product_ids)) != 100:
    raise ValueError('Expected 100 unique products, got %d' % len(set(product_ids)))

def col_float(df, col):
    return pd.to_numeric(df[col], errors='raise')
profit_per_unit = dict(zip(product_ids, col_float(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(product_ids, col_float(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(product_ids, col_float(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(product_ids, col_float(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(product_ids, pd.to_numeric(products_df['upper_demand_units'], errors='raise', downcast='integer')))
batch_size_units_set = set(pd.to_numeric(products_df['batch_size_units'], errors='raise', downcast='integer').tolist())
if len(batch_size_units_set) != 1:
    raise ValueError('batch_size_units is not constant across products')
batch_size_units = batch_size_units_set.pop()
if batch_size_units != 10:
    raise ValueError('batch_size_units is not 10 as expected')
resource_ids = ['R1', 'R2', 'R3']
resource_caps = {}
for r in resource_ids:
    cap_row = resources_df.loc[resources_df['resource'].str.casefold() == r.casefold()]
    if cap_row.empty:
        raise ValueError(f'Resource {r} not found in resources_capacities.csv')
    resource_caps[r] = float(cap_row.iloc[0]['capacity'])

def solve_problem():
    m = gp.Model('BatchProduction')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r1_per_unit[i] for i in product_ids)) <= resource_caps['R1'], name='r1_cap')
    m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r2_per_unit[i] for i in product_ids)) <= resource_caps['R2'], name='r2_cap')
    m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r3_per_unit[i] for i in product_ids)) <= resource_caps['R3'], name='r3_cap')
    for i in product_ids:
        m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')