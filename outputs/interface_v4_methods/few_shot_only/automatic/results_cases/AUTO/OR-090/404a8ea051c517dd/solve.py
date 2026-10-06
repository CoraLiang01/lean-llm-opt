import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
products = products_df['product'].astype(str).tolist()
resources = resources_df['resource'].astype(str).tolist()
batch_sizes = products_df['batch_size_units'].unique()
if len(batch_sizes) != 1:
    raise ValueError('All products must have the same batch_size_units.')
batch_size = int(batch_sizes[0])
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
resource_usage_per_unit = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(int).to_dict()
resource_capacities = resources_df.set_index('resource')['capacity'].astype(float).to_dict()

def solve_batch_production(products, resources, batch_size, profit_per_unit, resource_usage_per_unit, upper_demand_units, resource_capacities):
    m = gp.Model('BatchProductionPlanning')
    x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((batch_size * x[i] * profit_per_unit[i] for i in products)), gp.GRB.MAXIMIZE)
    for r in resources:
        m.addConstr(gp.quicksum((batch_size * x[i] * resource_usage_per_unit[r][i] for i in products)) <= resource_capacities[r], name=f'res_{r}')
    for i in products:
        m.addConstr(batch_size * x[i] <= upper_demand_units[i], name=f'demand_{i}')
    m.optimize()
    return m
m = solve_batch_production(products=products, resources=resources, batch_size=batch_size, profit_per_unit=profit_per_unit, resource_usage_per_unit=resource_usage_per_unit, upper_demand_units=upper_demand_units, resource_capacities=resource_capacities)