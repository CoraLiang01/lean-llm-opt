import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].astype(str).tolist()
n_products = len(product_ids)
batch_sizes = products_df['batch_size_units'].unique()
if len(batch_sizes) != 1 or batch_sizes[0] != 10:
    raise ValueError(f'batch_size_units must be constant 10, got {batch_sizes}')
batch_size_units = int(batch_sizes[0])
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(int).to_dict()
resource_ids = resources_df['resource'].astype(str).tolist()
resource_caps = resources_df.set_index('resource')['capacity'].astype(float).to_dict()
for r in ['R1', 'R2', 'R3']:
    if r not in resource_caps:
        raise ValueError(f'Resource {r} not found in resources_capacities.csv')
m = gp.Model('FactoryBatchProduction')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[i] * batch_size_units * profit_per_unit[i] for i in product_ids)), sense=gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * batch_size_units * r1_per_unit[i] for i in product_ids)) <= resource_caps['R1'], name='R1_capacity')
m.addConstr(gp.quicksum((x[i] * batch_size_units * r2_per_unit[i] for i in product_ids)) <= resource_caps['R2'], name='R2_capacity')
m.addConstr(gp.quicksum((x[i] * batch_size_units * r3_per_unit[i] for i in product_ids)) <= resource_caps['R3'], name='R3_capacity')
for i in product_ids:
    m.addConstr(x[i] * batch_size_units <= upper_demand_units[i], name=f'demand_ub_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_profit = m.objVal
    print(f'Optimal total profit: {total_profit:.2f}')
    print('Batch production plan (product: batches, units, profit):')
    for i in product_ids:
        batches = int(round(x[i].X))
        if batches > 0:
            units = batches * batch_size_units
            profit = units * profit_per_unit[i]
            print(f'  {i}: {batches} batches ({units} units), profit = {profit:.2f}')
    print('\nResource usage:')
    r1_used = sum((int(round(x[i].X)) * batch_size_units * r1_per_unit[i] for i in product_ids))
    r2_used = sum((int(round(x[i].X)) * batch_size_units * r2_per_unit[i] for i in product_ids))
    r3_used = sum((int(round(x[i].X)) * batch_size_units * r3_per_unit[i] for i in product_ids))
    print(f"  R1: {r1_used:.2f} / {resource_caps['R1']:.2f}")
    print(f"  R2: {r2_used:.2f} / {resource_caps['R2']:.2f}")
    print(f"  R3: {r3_used:.2f} / {resource_caps['R3']:.2f}")
else:
    print(f'No optimal solution found. Status: {m.status}')