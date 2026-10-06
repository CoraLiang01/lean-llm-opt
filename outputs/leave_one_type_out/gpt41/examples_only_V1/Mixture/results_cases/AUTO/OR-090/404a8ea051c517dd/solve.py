import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products = products_df['product'].astype(str).tolist()
resources = resources_df['resource'].astype(str).tolist()
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(int).to_dict()
batch_size_units_set = set(products_df['batch_size_units'].unique())
if len(batch_size_units_set) != 1:
    raise ValueError(f'Expected a single batch_size_units value, got {batch_size_units_set}')
batch_size_units = batch_size_units_set.pop()
if batch_size_units != 10:
    raise ValueError(f'batch_size_units must be 10, got {batch_size_units}')
resource_capacities = resources_df.set_index('resource')['capacity'].astype(float).to_dict()
for r in ['R1', 'R2', 'R3']:
    if r not in resource_capacities:
        raise KeyError(f'Resource {r} not found in resources_capacities.csv')
m = gp.Model('FactoryBatchProduction')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x[i] * profit_per_unit[i] for i in products)), sense=gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * batch_size_units * r1_per_unit[i] for i in products)) <= resource_capacities['R1'], name='res_R1')
m.addConstr(gp.quicksum((x[i] * batch_size_units * r2_per_unit[i] for i in products)) <= resource_capacities['R2'], name='res_R2')
m.addConstr(gp.quicksum((x[i] * batch_size_units * r3_per_unit[i] for i in products)) <= resource_capacities['R3'], name='res_R3')
for i in products:
    m.addConstr(x[i] * batch_size_units <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan (Batches) ---')
    for i in products:
        xi = x[i].X
        if xi > 1e-06:
            produced_units = int(round(xi * batch_size_units))
            print(f'  Product {i}: {int(round(xi))} batches ({produced_units} units)')
    print('\n--- Resource Usage ---')
    for r, per_unit in zip(['R1', 'R2', 'R3'], [r1_per_unit, r2_per_unit, r3_per_unit]):
        usage = sum((x[i].X * batch_size_units * per_unit[i] for i in products))
        print(f'  {r}: {usage:.2f} / {resource_capacities[r]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')