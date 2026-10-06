import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].astype(str).tolist()
n_products = len(product_ids)
batch_size_set = set(products_df['batch_size_units'].unique())
if len(batch_size_set) != 1 or 10 not in batch_size_set:
    raise ValueError('Batch size must be fixed at 10 for all products.')
batch_size = 10
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
resource_caps = resources_df.set_index('resource')['capacity'].to_dict()
for r in ['R1', 'R2', 'R3']:
    if r not in resource_caps:
        raise ValueError(f'Resource {r} not found in resources_capacities.csv.')

def solve_batch_production():
    m = gp.Model('BatchProductionPlanning')
    x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='x')
    m.setObjective(gp.quicksum((batch_size * x[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[i] * batch_size * r1_per_unit[i] for i in product_ids)) <= resource_caps['R1'], name='R1_capacity')
    m.addConstr(gp.quicksum((x[i] * batch_size * r2_per_unit[i] for i in product_ids)) <= resource_caps['R2'], name='R2_capacity')
    m.addConstr(gp.quicksum((x[i] * batch_size * r3_per_unit[i] for i in product_ids)) <= resource_caps['R3'], name='R3_capacity')
    for i in product_ids:
        m.addConstr(x[i] * batch_size <= upper_demand_units[i], name=f'demand_{i}')
    m.optimize()
    return m
m = solve_batch_production()