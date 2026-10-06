import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products = products_df['product'].astype(str).tolist()
resources = resources_df['resource'].astype(str).tolist()
batch_sizes = products_df.set_index('product')['batch_size_units'].to_dict()
if len(set(batch_sizes.values())) != 1:
    raise ValueError('Batch size is not uniform across products.')
batch_size = next(iter(batch_sizes.values()))
if batch_size != 10:
    raise ValueError('Batch size is not 10 as expected.')
profit_per_unit = products_df.set_index('product')['profit_per_unit'].to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].to_dict()
resource_usage_per_unit = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
upper_demand_units = products_df.set_index('product')['upper_demand_units'].to_dict()
resource_capacities = resources_df.set_index('resource')['capacity'].to_dict()
if set(products) != set(profit_per_unit.keys()):
    raise ValueError('Mismatch in product keys between index and profit_per_unit.')
if set(resources) != set(resource_capacities.keys()):
    raise ValueError('Mismatch in resource keys between index and resource_capacities.')
for r in resources:
    if r not in resource_usage_per_unit:
        raise ValueError(f'Missing resource usage data for resource {r}.')
m = gp.Model('BatchProductionPlanning')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[i] * batch_size * profit_per_unit[i] for i in products)), gp.GRB.MAXIMIZE)
for r in resources:
    usage = resource_usage_per_unit[r]
    m.addConstr(gp.quicksum((x[i] * batch_size * usage[i] for i in products)) <= resource_capacities[r], name=f'res_{r}')
for i in products:
    m.addConstr(x[i] * batch_size <= upper_demand_units[i], name=f'demand_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for i in products:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')