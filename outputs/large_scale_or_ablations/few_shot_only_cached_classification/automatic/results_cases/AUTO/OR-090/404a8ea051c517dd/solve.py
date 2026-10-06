import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].astype(str).tolist()
n_products = len(product_ids)
batch_size_set = set(products_df['batch_size_units'].unique())
if len(batch_size_set) != 1 or 10 not in batch_size_set:
    raise ValueError(f'All batch_size_units must be 10, got {batch_size_set}')
batch_size_units = 10
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
resource_ids = resources_df['resource'].astype(str).tolist()
if set(resource_ids) != {'R1', 'R2', 'R3'}:
    raise ValueError(f'Expected resources R1, R2, R3, got {resource_ids}')
resource_capacities = resources_df.set_index('resource')['capacity'].to_dict()
m = gp.Model('BatchProductionPlanning')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[i] * batch_size_units * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
resource_consumption = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
for r in ['R1', 'R2', 'R3']:
    m.addConstr(gp.quicksum((x[i] * batch_size_units * resource_consumption[r][i] for i in product_ids)) <= resource_capacities[r], name=f'res_{r}')
for i in product_ids:
    m.addConstr(x[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan (batches) ---')
    for i in product_ids:
        batches = int(round(x[i].X))
        if batches > 0:
            produced_units = batches * batch_size_units
            print(f'Product {i}: {batches} batches ({produced_units} units), Profit: {produced_units * profit_per_unit[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')