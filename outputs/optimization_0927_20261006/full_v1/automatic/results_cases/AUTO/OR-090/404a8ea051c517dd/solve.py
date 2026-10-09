import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', dtype=str, keep_default_na=False)
product_ids = products_df['product'].astype(str).tolist()
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(int).to_dict()
batch_size_units_set = set(products_df['batch_size_units'].astype(int).unique())
if len(batch_size_units_set) != 1:
    raise ValueError('All products must have the same batch_size_units per query.')
batch_size_units = batch_size_units_set.pop()
resource_ids = resources_df['resource'].astype(str).tolist()
capacity = resources_df.set_index('resource')['capacity'].astype(float).to_dict()
for r in ['R1', 'R2', 'R3']:
    if r not in capacity:
        raise KeyError(f'Resource {r} not found in resources_capacities.csv.')
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r1_per_unit[i] for i in product_ids)) <= capacity['R1'], name='resource_R1')
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r2_per_unit[i] for i in product_ids)) <= capacity['R2'], name='resource_R2')
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r3_per_unit[i] for i in product_ids)) <= capacity['R3'], name='resource_R3')
m.addConstrs((batch_size_units * x_vars[i] <= upper_demand_units[i] for i in product_ids), name='')
m.optimize()