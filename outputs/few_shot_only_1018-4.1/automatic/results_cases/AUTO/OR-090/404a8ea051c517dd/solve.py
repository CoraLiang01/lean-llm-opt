import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', dtype=str, keep_default_na=False)
resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', dtype=str, keep_default_na=False)
product_ids = [f'P{i}' for i in range(1, 101)]
if not set(product_ids).issubset(set(products_df['product'].str.strip())):
    missing = set(product_ids) - set(products_df['product'].str.strip())
    raise ValueError(f'Missing products in CSV: {missing}')
products_df = products_df.set_index(products_df['product'].str.strip())
batch_size_units = int(products_df.loc[product_ids[0], 'batch_size_units'])
profit_per_unit = {pid: float(products_df.loc[pid, 'profit_per_unit']) for pid in product_ids}
r1_per_unit = {pid: float(products_df.loc[pid, 'r1_per_unit']) for pid in product_ids}
r2_per_unit = {pid: float(products_df.loc[pid, 'r2_per_unit']) for pid in product_ids}
r3_per_unit = {pid: float(products_df.loc[pid, 'r3_per_unit']) for pid in product_ids}
upper_demand_units = {pid: int(products_df.loc[pid, 'upper_demand_units']) for pid in product_ids}
resources_df = resources_df.set_index(resources_df['resource'].str.strip())
resource_ids = ['R1', 'R2', 'R3']
if not set(resource_ids).issubset(set(resources_df.index)):
    missing = set(resource_ids) - set(resources_df.index)
    raise ValueError(f'Missing resources in CSV: {missing}')
resource_capacities = {rid: float(resources_df.loc[rid, 'capacity']) for rid in resource_ids}
m = gp.Model('BatchProductionPlanning')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((batch_size_units * x_vars[pid] * profit_per_unit[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((batch_size_units * x_vars[pid] * r1_per_unit[pid] for pid in product_ids)) <= resource_capacities['R1'], name='resource_R1')
m.addConstr(gp.quicksum((batch_size_units * x_vars[pid] * r2_per_unit[pid] for pid in product_ids)) <= resource_capacities['R2'], name='resource_R2')
m.addConstr(gp.quicksum((batch_size_units * x_vars[pid] * r3_per_unit[pid] for pid in product_ids)) <= resource_capacities['R3'], name='resource_R3')
for pid in product_ids:
    m.addConstr(x_vars[pid] * batch_size_units <= upper_demand_units[pid], name=f'demand_{pid}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_profit = m.objVal
    print(f'Optimal total profit: {total_profit:.2f}')
    print('--- Production Plan (batches, units, profit) ---')
    for pid in product_ids:
        batches = int(round(x_vars[pid].X))
        units = batches * batch_size_units
        if batches > 0:
            profit = units * profit_per_unit[pid]
            print(f'{pid}: {batches} batches ({units} units), profit = {profit:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')