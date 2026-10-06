import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].astype(str).tolist()
batch_sizes = products_df.set_index('product')['batch_size_units']
if not (batch_sizes == batch_sizes.iloc[0]).all():
    raise ValueError('Batch size is not constant across all products.')
batch_size_units = int(batch_sizes.iloc[0])
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
resource_caps = resources_df.set_index('resource')['capacity'].to_dict()
for res in ['R1', 'R2', 'R3']:
    if res not in resource_caps:
        raise ValueError(f'Resource {res} not found in resources_capacities.csv.')
m = gp.Model('FactoryBatchProduction')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[i] * batch_size_units * profit_per_unit[i] for i in product_ids)), sense=gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * batch_size_units * r1_per_unit[i] for i in product_ids)) <= resource_caps['R1'], name='R1_capacity')
m.addConstr(gp.quicksum((x[i] * batch_size_units * r2_per_unit[i] for i in product_ids)) <= resource_caps['R2'], name='R2_capacity')
m.addConstr(gp.quicksum((x[i] * batch_size_units * r3_per_unit[i] for i in product_ids)) <= resource_caps['R3'], name='R3_capacity')
for i in product_ids:
    m.addConstr(x[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()