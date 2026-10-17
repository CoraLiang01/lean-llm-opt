import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
products_df['product'] = products_df['product'].str.strip()
product_ids = products_df['product'].tolist()

def to_float_col(df, col):
    return df[col].astype(float)

def to_int_col(df, col):
    return df[col].astype(int)
profit_per_unit = to_float_col(products_df, 'profit_per_unit')
r1_per_unit = to_float_col(products_df, 'r1_per_unit')
r2_per_unit = to_float_col(products_df, 'r2_per_unit')
r3_per_unit = to_float_col(products_df, 'r3_per_unit')
upper_demand_units = to_int_col(products_df, 'upper_demand_units')
batch_size_units = to_int_col(products_df, 'batch_size_units')
batch_size_dict = dict(zip(product_ids, batch_size_units))
if len(set(batch_size_dict.values())) != 1:
    raise ValueError('Batch size is not uniform across products.')
batch_size = batch_size_units.iloc[0]
profit_per_unit_dict = dict(zip(product_ids, profit_per_unit))
r1_per_unit_dict = dict(zip(product_ids, r1_per_unit))
r2_per_unit_dict = dict(zip(product_ids, r2_per_unit))
r3_per_unit_dict = dict(zip(product_ids, r3_per_unit))
upper_demand_units_dict = dict(zip(product_ids, upper_demand_units))
resources_df['resource'] = resources_df['resource'].str.strip()
resources_df['resource_norm'] = resources_df['resource'].str.strip().str.casefold()
resource_ids = resources_df['resource'].tolist()
resource_norm_map = dict(zip(resources_df['resource_norm'], resources_df['resource']))
resource_capacity_dict = {}
for idx, row in resources_df.iterrows():
    resource = row['resource']
    capacity = float(row['capacity'])
    resource_capacity_dict[resource] = capacity
resource_field_to_id = {'r1_per_unit': None, 'r2_per_unit': None, 'r3_per_unit': None}
for field, norm in zip(['r1_per_unit', 'r2_per_unit', 'r3_per_unit'], ['r1', 'r2', 'r3']):
    if norm in resource_norm_map:
        resource_field_to_id[field] = resource_norm_map[norm]
    else:
        raise KeyError(f"Resource '{norm}' not found in resources_capacities.csv.")
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size * x_vars[i] * profit_per_unit_dict[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((batch_size * x_vars[i] * r1_per_unit_dict[i] for i in product_ids)) <= resource_capacity_dict[resource_field_to_id['r1_per_unit']], name='resource_R1')
m.addConstr(gp.quicksum((batch_size * x_vars[i] * r2_per_unit_dict[i] for i in product_ids)) <= resource_capacity_dict[resource_field_to_id['r2_per_unit']], name='resource_R2')
m.addConstr(gp.quicksum((batch_size * x_vars[i] * r3_per_unit_dict[i] for i in product_ids)) <= resource_capacity_dict[resource_field_to_id['r3_per_unit']], name='resource_R3')
m.addConstrs((batch_size * x_vars[i] <= upper_demand_units_dict[i] for i in product_ids), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan (batches) ---')
    for i in product_ids:
        batches = int(round(x_vars[i].X))
        if batches > 0:
            produced_units = batches * batch_size
            print(f'Product {i}: {batches} batches ({produced_units} units)')
else:
    print(f'No optimal solution found. Status: {m.status}')