import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].astype(str).tolist()
resource_ids = resources_df['resource'].astype(str).tolist()
batch_sizes = products_df.set_index('product')['batch_size_units'].astype(int).to_dict()
if len(set(batch_sizes.values())) != 1:
    raise ValueError('Batch size is not consistent across all products.')
batch_size = next(iter(batch_sizes.values()))
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(int).to_dict()
resource_capacities = resources_df.set_index('resource')['capacity'].astype(float).to_dict()
for r in ['R1', 'R2', 'R3']:
    if r not in resource_capacities:
        raise ValueError(f'Resource {r} not found in resource capacities.')
m = gp.Model('BatchProductionPlanning')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size * x[i] * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * batch_size * r1_per_unit[i] for i in product_ids)) <= resource_capacities['R1'], name='r1_cap')
m.addConstr(gp.quicksum((x[i] * batch_size * r2_per_unit[i] for i in product_ids)) <= resource_capacities['R2'], name='r2_cap')
m.addConstr(gp.quicksum((x[i] * batch_size * r3_per_unit[i] for i in product_ids)) <= resource_capacities['R3'], name='r3_cap')
for i in product_ids:
    m.addConstr(x[i] * batch_size <= upper_demand_units[i], name=f'demand_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for i in product_ids:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')