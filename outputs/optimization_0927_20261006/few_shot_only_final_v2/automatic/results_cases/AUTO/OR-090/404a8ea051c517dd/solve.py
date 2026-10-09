import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
product_ids = products_df['product'].tolist()
batch_size_col = 'batch_size_units'
if not all(products_df[batch_size_col] == products_df[batch_size_col].iloc[0]):
    raise ValueError('Batch size is not consistent across all products.')
batch_size_units = int(products_df[batch_size_col].iloc[0])
profit_per_unit = {pid: float(products_df.loc[products_df['product'] == pid, 'profit_per_unit'].values[0]) for pid in product_ids}
r1_per_unit = {pid: float(products_df.loc[products_df['product'] == pid, 'r1_per_unit'].values[0]) for pid in product_ids}
r2_per_unit = {pid: float(products_df.loc[products_df['product'] == pid, 'r2_per_unit'].values[0]) for pid in product_ids}
r3_per_unit = {pid: float(products_df.loc[products_df['product'] == pid, 'r3_per_unit'].values[0]) for pid in product_ids}
upper_demand_units = {pid: int(float(products_df.loc[products_df['product'] == pid, 'upper_demand_units'].values[0])) for pid in product_ids}
resource_ids = resources_df['resource'].tolist()
capacity = {}
for (idx, row) in resources_df.iterrows():
    rid = str(row['resource'])
    capacity[rid] = float(row['capacity'])
resource_per_unit = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
for rid in resource_ids:
    if rid not in resource_per_unit:
        raise ValueError(f'Resource {rid} not found in per-unit consumption columns.')
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[pid] * profit_per_unit[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
for rid in resource_ids:
    m.addConstr(gp.quicksum((batch_size_units * x_vars[pid] * resource_per_unit[rid][pid] for pid in product_ids)) <= capacity[rid], name=f'res_{rid}')
for pid in product_ids:
    m.addConstr(batch_size_units * x_vars[pid] <= upper_demand_units[pid], name=f'demand_{pid}')
m.optimize()