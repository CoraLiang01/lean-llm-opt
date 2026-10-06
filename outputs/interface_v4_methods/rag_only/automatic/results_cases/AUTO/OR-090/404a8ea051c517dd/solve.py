import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products_df['product'] = products_df['product'].astype(str).str.strip()
resources_df['resource'] = resources_df['resource'].astype(str).str.strip()
product_ids = products_df['product'].tolist()
resource_ids = resources_df['resource'].tolist()
batch_size_units = int(products_df['batch_size_units'].iloc[0])
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
capacity = resources_df.set_index('resource')['capacity'].to_dict()
if set(product_ids) != set(profit_per_unit.keys()):
    raise ValueError('Mismatch in product identifiers between index set and profit_per_unit data.')
if set(resource_ids) != {'R1', 'R2', 'R3'}:
    raise ValueError('Resource identifiers in resources_capacities.csv must be exactly R1, R2, R3.')
m = gp.Model('factory_batch_production')
x = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, obj=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x[i] * profit_per_unit[i] for i in product_ids)), GRB.MAXIMIZE)
for r in resource_ids:
    if r == 'R1':
        m.addConstr(gp.quicksum((batch_size_units * x[i] * r1_per_unit[i] for i in product_ids)) <= capacity[r], name=f'res_{r}')
    elif r == 'R2':
        m.addConstr(gp.quicksum((batch_size_units * x[i] * r2_per_unit[i] for i in product_ids)) <= capacity[r], name=f'res_{r}')
    elif r == 'R3':
        m.addConstr(gp.quicksum((batch_size_units * x[i] * r3_per_unit[i] for i in product_ids)) <= capacity[r], name=f'res_{r}')
    else:
        raise ValueError(f'Unexpected resource identifier: {r}')
for i in product_ids:
    m.addConstr(batch_size_units * x[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()