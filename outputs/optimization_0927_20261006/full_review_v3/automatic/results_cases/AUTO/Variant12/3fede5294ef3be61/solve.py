import gurobipy as gp
import pandas as pd
import numpy as np
import re
plant_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv'
retailer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv'
plant_capacity_df = pd.read_csv(plant_capacity_path, dtype=str, keep_default_na=False)
retailer_demand_df = pd.read_csv(retailer_demand_path, dtype=str, keep_default_na=False)
route_variable_costs_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
plants = plant_capacity_df['Plant'].str.strip().tolist()
retailers = retailer_demand_df['Retailer'].str.strip().tolist()
plant_capacity_df['SupplyCapacity'] = plant_capacity_df['SupplyCapacity'].astype(int)
plant_capacity = dict(zip(plant_capacity_df['Plant'].str.strip(), plant_capacity_df['SupplyCapacity']))
retailer_demand_df['Demand'] = retailer_demand_df['Demand'].astype(int)
retailer_demand = dict(zip(retailer_demand_df['Retailer'].str.strip(), retailer_demand_df['Demand']))
route_variable_costs = {}
for (_, row) in route_variable_costs_df.iterrows():
    i = row['Plant'].strip()
    for j in retailers:
        route_variable_costs[i, j] = int(row[j])
route_fixed_costs = {}
for (_, row) in route_fixed_costs_df.iterrows():
    i = row['Plant'].strip()
    for j in retailers:
        route_fixed_costs[i, j] = int(row[j])
big_m = {}
for i in plants:
    for j in retailers:
        big_m[i, j] = min(plant_capacity[i], retailer_demand[j])
for i in plants:
    if i not in plant_capacity:
        raise ValueError(f'Missing plant capacity for plant {i}')
for j in retailers:
    if j not in retailer_demand:
        raise ValueError(f'Missing retailer demand for retailer {j}')
for i in plants:
    for j in retailers:
        if (i, j) not in route_variable_costs:
            raise ValueError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in route_fixed_costs:
            raise ValueError(f'Missing fixed cost for route ({i},{j})')
        if (i, j) not in big_m:
            raise ValueError(f'Missing Big-M for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_variable_costs[i, j] * x_vars[i, j] + route_fixed_costs[i, j] * y_vars[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == retailer_demand[j])
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in retailers)) <= plant_capacity[i])
for i in plants:
    for j in retailers:
        m.addConstr(x_vars[i, j] <= big_m[i, j] * y_vars[i, j])
m.optimize()