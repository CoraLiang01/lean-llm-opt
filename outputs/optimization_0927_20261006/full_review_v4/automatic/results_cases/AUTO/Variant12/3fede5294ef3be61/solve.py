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
plants = plant_capacity_df['Plant'].astype(str).str.strip().tolist()
retailers = retailer_demand_df['Retailer'].astype(str).str.strip().tolist()
plant_capacity_df['SupplyCapacity'] = plant_capacity_df['SupplyCapacity'].astype(int)
plant_capacity = dict(zip(plant_capacity_df['Plant'].astype(str).str.strip(), plant_capacity_df['SupplyCapacity']))
retailer_demand_df['Demand'] = retailer_demand_df['Demand'].astype(int)
retailer_demand = dict(zip(retailer_demand_df['Retailer'].astype(str).str.strip(), retailer_demand_df['Demand']))
route_variable_costs = {}
for (_, row) in route_variable_costs_df.iterrows():
    plant = str(row['Plant']).strip()
    for retailer in retailers:
        cost_val = row[retailer]
        if cost_val == '':
            raise ValueError(f'Missing variable cost for Plant {plant}, Retailer {retailer}')
        route_variable_costs[plant, retailer] = int(cost_val)
route_fixed_costs = {}
for (_, row) in route_fixed_costs_df.iterrows():
    plant = str(row['Plant']).strip()
    for retailer in retailers:
        cost_val = row[retailer]
        if cost_val == '':
            raise ValueError(f'Missing fixed cost for Plant {plant}, Retailer {retailer}')
        route_fixed_costs[plant, retailer] = int(cost_val)
big_m = {}
for i in plants:
    for j in retailers:
        if i not in plant_capacity:
            raise KeyError(f'Plant {i} missing from plant_capacity')
        if j not in retailer_demand:
            raise KeyError(f'Retailer {j} missing from retailer_demand')
        big_m[i, j] = min(plant_capacity[i], retailer_demand[j])
for i in plants:
    for j in retailers:
        if (i, j) not in route_variable_costs:
            raise KeyError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in route_fixed_costs:
            raise KeyError(f'Missing fixed cost for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_variable_costs[i, j] * x_vars[i, j] + route_fixed_costs[i, j] * y_vars[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == retailer_demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in retailers)) <= plant_capacity[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x_vars[i, j] <= big_m[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()