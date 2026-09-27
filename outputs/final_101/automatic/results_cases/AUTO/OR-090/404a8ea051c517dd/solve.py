import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].astype(str).tolist()
batch_size_units = products_df['batch_size_units'].iloc[0]
if not np.all(products_df['batch_size_units'] == batch_size_units):
    raise ValueError('All products must have the same batch_size_units.')
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(int).to_dict()
resource_ids = resources_df['resource'].astype(str).tolist()
resource_capacities = resources_df.set_index('resource')['capacity'].astype(float).to_dict()
resource_usage_per_unit = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
m = gp.Model('BatchProductionPlanning')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[i] * batch_size_units * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
for r in resource_ids:
    usage_dict = resource_usage_per_unit[r]
    m.addConstr(gp.quicksum((x[i] * batch_size_units * usage_dict[i] for i in product_ids)) <= resource_capacities[r], name=f'res_{r}')
for i in product_ids:
    m.addConstr(x[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan (batches) ---')
    for i in product_ids:
        batches = int(round(x[i].X))
        if batches > 0:
            print(f'Product {i}: {batches} batches ({batches * batch_size_units} units)')
else:
    print(f'No optimal solution found. Status: {m.status}')