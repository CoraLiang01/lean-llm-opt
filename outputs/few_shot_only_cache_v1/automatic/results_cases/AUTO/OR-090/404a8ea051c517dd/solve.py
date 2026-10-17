import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', dtype=str, keep_default_na=False)
product_id_col = 'product'
products = products_df[product_id_col].astype(str).tolist()
resource_id_col = 'resource'
resources = resources_df[resource_id_col].astype(str).tolist()
batch_size_col = 'batch_size_units'
batch_size_units_dict = products_df.set_index(product_id_col)[batch_size_col].astype(int).to_dict()
batch_sizes = set(batch_size_units_dict.values())
if len(batch_sizes) != 1:
    raise ValueError(f'Batch size units are not uniform across products: {batch_sizes}')
batch_size_units = batch_sizes.pop()
profit_per_unit = products_df.set_index(product_id_col)['profit_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index(product_id_col)['upper_demand_units'].astype(int).to_dict()
resource_per_unit = {}
for r in resources:
    colname = f'{r.lower()}_per_unit'
    if colname not in products_df.columns:
        raise KeyError(f'Missing column {colname} in products data')
    resource_per_unit[r] = products_df.set_index(product_id_col)[colname].astype(float).to_dict()
capacity = resources_df.set_index(resource_id_col)['capacity'].astype(float).to_dict()
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[i] * batch_size_units * profit_per_unit[i] for i in products)), gp.GRB.MAXIMIZE)
for r in resources:
    m.addConstr(gp.quicksum((x_vars[i] * batch_size_units * resource_per_unit[r][i] for i in products)) <= capacity[r], name=f'res_{r}')
for i in products:
    m.addConstr(x_vars[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()