import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products_df['product'] = products_df['product'].astype(str).str.strip()
resources_df['resource'] = resources_df['resource'].astype(str).str.strip()
products = products_df['product'].tolist()
resources = resources_df['resource'].tolist()
if not (products_df['batch_size_units'] == 10).all():
    raise ValueError('All batch_size_units must be 10 for all products.')
batch_size_units = 10
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
resource_consumption = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
capacity = resources_df.set_index('resource')['capacity'].to_dict()
for p in products:
    if p not in profit_per_unit or p not in upper_demand_units or p not in r1_per_unit or (p not in r2_per_unit) or (p not in r3_per_unit):
        raise ValueError(f'Missing parameter data for product {p}')
for r in resources:
    if r not in capacity:
        raise ValueError(f'Missing capacity for resource {r}')
m = gp.Model('factory_batch_production')
m.setParam('MIPGap', 0.0001)
x = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[p] * batch_size_units * profit_per_unit[p] for p in products)), GRB.MAXIMIZE)
for r in resources:
    m.addConstr(gp.quicksum((x[p] * batch_size_units * resource_consumption[r][p] for p in products)) <= capacity[r], name='')
for p in products:
    m.addConstr(x[p] * batch_size_units <= upper_demand_units[p], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for p in products:
        var = x[p]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')