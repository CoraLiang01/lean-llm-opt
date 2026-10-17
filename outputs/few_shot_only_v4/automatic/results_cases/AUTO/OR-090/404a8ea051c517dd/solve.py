import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',', dtype=str, keep_default_na=False)
product_ids = [f'P{i}' for i in range(1, 101)]
if not set(product_ids).issubset(set(products_df['product_id'])):
    missing = set(product_ids) - set(products_df['product_id'])
    raise ValueError(f'Missing products in CSV: {missing}')
products_df = products_df.set_index('product_id', drop=False)

def get_numeric_col(df, col, ids, dtype=float):
    vals = {}
    for pid in ids:
        val = df.at[pid, col]
        try:
            vals[pid] = dtype(val)
        except Exception:
            raise ValueError(f'Invalid value for {col} of product {pid}: {val}')
    return vals
profit_per_unit = get_numeric_col(products_df, 'profit_per_unit', product_ids, float)
r1_per_unit = get_numeric_col(products_df, 'r1_per_unit', product_ids, float)
r2_per_unit = get_numeric_col(products_df, 'r2_per_unit', product_ids, float)
r3_per_unit = get_numeric_col(products_df, 'r3_per_unit', product_ids, float)
upper_demand_units = get_numeric_col(products_df, 'upper_demand_units', product_ids, int)
resource_ids = ['R1', 'R2', 'R3']
resource_col_map = {'R1': 'r1_per_unit', 'R2': 'r2_per_unit', 'R3': 'r3_per_unit'}
if not set(resource_ids).issubset(set(resources_df['resource_id'])):
    missing = set(resource_ids) - set(resources_df['resource_id'])
    raise ValueError(f'Missing resources in CSV: {missing}')
resources_df = resources_df.set_index('resource_id', drop=False)
resource_capacities = {}
for rid in resource_ids:
    val = resources_df.at[rid, 'capacity']
    try:
        resource_capacities[rid] = float(val)
    except Exception:
        raise ValueError(f'Invalid capacity for resource {rid}: {val}')
batch_size_units = 10
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[pid] * profit_per_unit[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
for rid in resource_ids:
    if rid == 'R1':
        per_unit = r1_per_unit
    elif rid == 'R2':
        per_unit = r2_per_unit
    elif rid == 'R3':
        per_unit = r3_per_unit
    else:
        raise ValueError(f'Unexpected resource id: {rid}')
    m.addConstr(gp.quicksum((batch_size_units * x_vars[pid] * per_unit[pid] for pid in product_ids)) <= resource_capacities[rid], name=f'res_{rid}')
for pid in product_ids:
    m.addConstr(batch_size_units * x_vars[pid] <= upper_demand_units[pid], name=f'demand_{pid}')
m.optimize()