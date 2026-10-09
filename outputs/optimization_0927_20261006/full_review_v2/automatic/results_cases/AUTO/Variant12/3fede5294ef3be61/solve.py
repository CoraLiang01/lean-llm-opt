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

def norm_str(s):
    return s.strip().casefold()
plant_capacity_df['Plant'] = plant_capacity_df['Plant'].apply(norm_str)
retailer_demand_df['Retailer'] = retailer_demand_df['Retailer'].apply(norm_str)
route_variable_costs_df['Plant'] = route_variable_costs_df['Plant'].apply(norm_str)
route_fixed_costs_df['Plant'] = route_fixed_costs_df['Plant'].apply(norm_str)
plants = list(plant_capacity_df['Plant'])
retailers = list(retailer_demand_df['Retailer'])
plant_capacity = {}
for (_, row) in plant_capacity_df.iterrows():
    plant = row['Plant']
    try:
        cap = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f"Invalid SupplyCapacity for plant {plant}: {row['SupplyCapacity']}")
    plant_capacity[plant] = cap
retailer_demand = {}
for (_, row) in retailer_demand_df.iterrows():
    retailer = row['Retailer']
    try:
        dem = int(row['Demand'])
    except Exception:
        raise ValueError(f"Invalid Demand for retailer {retailer}: {row['Demand']}")
    retailer_demand[retailer] = dem
route_variable_cost = {}
for (_, row) in route_variable_costs_df.iterrows():
    plant = row['Plant']
    for retailer in retailers:
        if retailer not in row:
            raise KeyError(f'Retailer {retailer} not found in route_variable_costs.csv columns')
        try:
            cost = int(row[retailer])
        except Exception:
            raise ValueError(f'Invalid variable cost for route ({plant},{retailer}): {row[retailer]}')
        route_variable_cost[plant, retailer] = cost
route_fixed_cost = {}
for (_, row) in route_fixed_costs_df.iterrows():
    plant = row['Plant']
    for retailer in retailers:
        if retailer not in row:
            raise KeyError(f'Retailer {retailer} not found in route_fixed_costs.csv columns')
        try:
            cost = int(row[retailer])
        except Exception:
            raise ValueError(f'Invalid fixed cost for route ({plant},{retailer}): {row[retailer]}')
        route_fixed_cost[plant, retailer] = cost
bigM = {}
for i in plants:
    for j in retailers:
        if i not in plant_capacity:
            raise KeyError(f'Plant {i} missing from plant_capacity')
        if j not in retailer_demand:
            raise KeyError(f'Retailer {j} missing from retailer_demand')
        bigM[i, j] = min(plant_capacity[i], retailer_demand[j])
for i in plants:
    for j in retailers:
        if (i, j) not in route_variable_cost:
            raise KeyError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in route_fixed_cost:
            raise KeyError(f'Missing fixed cost for route ({i},{j})')
        if (i, j) not in bigM:
            raise KeyError(f'Missing Big-M for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_variable_cost[i, j] * x_vars[i, j] + route_fixed_cost[i, j] * y_vars[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == retailer_demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in retailers)) <= plant_capacity[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x_vars[i, j] <= bigM[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()