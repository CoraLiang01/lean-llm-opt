import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', dtype=str, keep_default_na=False)
products_df['product'] = products_df['product'].str.strip()
products = products_df['product'].tolist()

def to_float_col(df, col):
    return pd.to_numeric(df[col], errors='raise')

def to_int_col(df, col):
    return pd.to_numeric(df[col], errors='raise', downcast='integer')
profit_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'profit_per_unit')))
r1_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'r1_per_unit')))
r2_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'r2_per_unit')))
r3_per_unit = dict(zip(products_df['product'], to_float_col(products_df, 'r3_per_unit')))
upper_demand_units = dict(zip(products_df['product'], to_int_col(products_df, 'upper_demand_units')))
batch_size_units_set = set(to_int_col(products_df, 'batch_size_units'))
if len(batch_size_units_set) != 1:
    raise ValueError('batch_size_units is not constant across products.')
batch_size_units = batch_size_units_set.pop()
if batch_size_units != 10:
    raise ValueError('batch_size_units is not 10 as expected.')
resources_df['resource'] = resources_df['resource'].str.strip()
resource_ids = ['R1', 'R2', 'R3']
resource_capacities = {}
for r in resource_ids:
    cap_row = resources_df[resources_df['resource'].str.casefold() == r.casefold()]
    if cap_row.empty:
        raise ValueError(f'Resource {r} not found in resources_capacities.csv')
    resource_capacities[r] = float(cap_row.iloc[0]['capacity'])
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[i] * profit_per_unit[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r1_per_unit[i] for i in products)) <= resource_capacities['R1'], name='resource_R1')
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r2_per_unit[i] for i in products)) <= resource_capacities['R2'], name='resource_R2')
m.addConstr(gp.quicksum((batch_size_units * x_vars[i] * r3_per_unit[i] for i in products)) <= resource_capacities['R3'], name='resource_R3')
for i in products:
    m.addConstr(batch_size_units * x_vars[i] <= upper_demand_units[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Production Plan (batches) ---')
    for i in products:
        batches = int(round(x_vars[i].X))
        units = batches * batch_size_units
        print(f'{i}: {batches} batches ({units} units), profit/unit={profit_per_unit[i]:.2f}, demand cap={upper_demand_units[i]}')
    print('--- Resource Usage ---')
    r1_used = sum((batch_size_units * x_vars[i].X * r1_per_unit[i] for i in products))
    r2_used = sum((batch_size_units * x_vars[i].X * r2_per_unit[i] for i in products))
    r3_used = sum((batch_size_units * x_vars[i].X * r3_per_unit[i] for i in products))
    print(f"R1: {r1_used:.2f} / {resource_capacities['R1']:.2f}")
    print(f"R2: {r2_used:.2f} / {resource_capacities['R2']:.2f}")
    print(f"R3: {r3_used:.2f} / {resource_capacities['R3']:.2f}")
else:
    print(f'No optimal solution found. Status: {m.status}')