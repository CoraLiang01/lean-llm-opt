import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products_df['product'] = products_df['product'].astype(str).str.strip()
resources_df['resource'] = resources_df['resource'].astype(str).str.strip()
product_ids = products_df['product'].tolist()
resource_ids = resources_df['resource'].tolist()
batch_sizes = products_df['batch_size_units'].unique()
if len(batch_sizes) != 1 or batch_sizes[0] != 10:
    raise ValueError('batch_size_units must be 10 for all products.')
batch_size_units = int(batch_sizes[0])
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
resource_capacities = resources_df.set_index('resource')['capacity'].to_dict()
for r in ['R1', 'R2', 'R3']:
    if r not in resource_capacities:
        raise ValueError(f'Resource {r} not found in resources_capacities.csv.')
m = gp.Model('BatchProductionPlanning')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * profit_per_unit[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((r1_per_unit[i] * batch_size_units * x[i] for i in product_ids)) <= resource_capacities['R1'], name='res_R1')
m.addConstr(gp.quicksum((r2_per_unit[i] * batch_size_units * x[i] for i in product_ids)) <= resource_capacities['R2'], name='res_R2')
m.addConstr(gp.quicksum((r3_per_unit[i] * batch_size_units * x[i] for i in product_ids)) <= resource_capacities['R3'], name='res_R3')
for i in product_ids:
    m.addConstr(batch_size_units * x[i] <= upper_demand_units[i], name=f'demand_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')