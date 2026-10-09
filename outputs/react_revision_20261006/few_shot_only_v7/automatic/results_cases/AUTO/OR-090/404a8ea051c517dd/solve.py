import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
product_keys = products_df['product'].tolist()
if len(set(product_keys)) != len(product_keys):
    raise ValueError('Duplicate product identifiers found in factory_products_100.csv')

def to_float_col(df, col):
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")

def to_int_col(df, col):
    try:
        return df[col].astype(int)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to int: {e}")
profit_per_unit = dict(zip(product_keys, to_float_col(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(product_keys, to_float_col(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(product_keys, to_float_col(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(product_keys, to_float_col(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(product_keys, to_int_col(products_df, 'upper_demand_units')))
batch_size_units_set = set(to_int_col(products_df, 'batch_size_units'))
if len(batch_size_units_set) != 1:
    raise ValueError('batch_size_units is not constant across all products.')
batch_size_units = batch_size_units_set.pop()
if batch_size_units <= 0:
    raise ValueError('batch_size_units must be positive.')
resource_keys = resources_df['resource'].tolist()
if len(set(resource_keys)) != len(resource_keys):
    raise ValueError('Duplicate resource identifiers found in resources_capacities.csv')
capacity = {}
for (idx, row) in resources_df.iterrows():
    r = row['resource']
    try:
        cap = float(row['capacity'])
    except Exception as e:
        raise ValueError(f"Capacity for resource '{r}' could not be converted to float: {e}")
    capacity[r] = cap
required_resources = ['R1', 'R2', 'R3']
for r in required_resources:
    if r not in capacity:
        raise ValueError(f"Resource '{r}' not found in resources_capacities.csv")

def solve_problem():
    m = gp.Model('BatchProductionPlanning')
    x_vars = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[i] * batch_size_units * profit_per_unit[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r1_per_unit[i] for i in product_keys)) <= capacity['R1'], name='r1_cap')
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r2_per_unit[i] for i in product_keys)) <= capacity['R2'], name='r2_cap')
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * r3_per_unit[i] for i in product_keys)) <= capacity['R3'], name='r3_cap')
    for i in product_keys:
        m.addConstr(x_vars[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')