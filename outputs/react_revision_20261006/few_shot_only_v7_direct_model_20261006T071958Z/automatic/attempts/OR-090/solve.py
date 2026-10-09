import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
product_ids = products_df['product'].tolist()
resource_ids = resources_df['resource'].tolist()

def to_float_series(df, col, key_col):
    try:
        return pd.Series(df[col], index=df[key_col]).astype(float).to_dict()
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to float: {e}")

def to_int_series(df, col, key_col):
    try:
        return pd.Series(df[col], index=df[key_col]).astype(int).to_dict()
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to int: {e}")
profit_per_unit = to_float_series(products_df, 'profit_per_unit', 'product')
r1_per_unit = to_float_series(products_df, 'r1_per_unit', 'product')
r2_per_unit = to_float_series(products_df, 'r2_per_unit', 'product')
r3_per_unit = to_float_series(products_df, 'r3_per_unit', 'product')
upper_demand_units = to_int_series(products_df, 'upper_demand_units', 'product')
batch_size_units_dict = to_int_series(products_df, 'batch_size_units', 'product')
batch_sizes = set(batch_size_units_dict.values())
if len(batch_sizes) != 1:
    raise ValueError(f'Batch size is not unique across products: {batch_sizes}')
batch_size_units = batch_sizes.pop()
if batch_size_units != 10:
    raise ValueError(f'Batch size is not 10 as expected, got {batch_size_units}')
resource_capacities = to_float_series(resources_df, 'capacity', 'resource')
for pid in product_ids:
    if pid not in profit_per_unit or pid not in r1_per_unit or pid not in r2_per_unit or (pid not in r3_per_unit) or (pid not in upper_demand_units):
        raise KeyError(f'Missing parameter(s) for product {pid}')
for rid in ['R1', 'R2', 'R3']:
    if rid not in resource_capacities:
        raise KeyError(f'Missing capacity for resource {rid}')

def solve_problem():
    m = gp.Model('BatchProduction')
    x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[i] * batch_size_units * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r1_per_unit[i] for i in product_ids)) <= resource_capacities['R1'], name='r1_cap')
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r2_per_unit[i] for i in product_ids)) <= resource_capacities['R2'], name='r2_cap')
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r3_per_unit[i] for i in product_ids)) <= resource_capacities['R3'], name='r3_cap')
    m.addConstrs((x_vars[i] * batch_size_units <= upper_demand_units[i] for i in product_ids), name='')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')