import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].astype(str).tolist()
n_products = len(product_ids)
batch_sizes = products_df['batch_size_units'].unique()
if len(batch_sizes) != 1 or batch_sizes[0] != 10:
    raise ValueError('batch_size_units must be 10 for all products.')
batch_size_units = int(batch_sizes[0])
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
resource_ids = resources_df['resource'].astype(str).tolist()
resource_caps = resources_df.set_index('resource')['capacity'].to_dict()
for r in ['R1', 'R2', 'R3']:
    if r not in resource_caps:
        raise ValueError(f'Resource {r} not found in resources_capacities.csv.')

def solve_problem():
    m = gp.Model('BatchProduction')
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[i] * batch_size_units * profit_per_unit[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[i] * batch_size_units * r1_per_unit[i] for i in product_ids)) <= resource_caps['R1'], name='r1_cap')
    m.addConstr(gp.quicksum((x[i] * batch_size_units * r2_per_unit[i] for i in product_ids)) <= resource_caps['R2'], name='r2_cap')
    m.addConstr(gp.quicksum((x[i] * batch_size_units * r3_per_unit[i] for i in product_ids)) <= resource_caps['R3'], name='r3_cap')
    for i in product_ids:
        m.addConstr(x[i] * batch_size_units <= upper_demand_units[i], name='demand_' + i)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')