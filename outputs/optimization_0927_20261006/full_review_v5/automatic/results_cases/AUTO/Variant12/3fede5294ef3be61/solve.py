import gurobipy as gp
import pandas as pd
import numpy as np
import re
plant_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv'
retailer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv'
plant_capacity_df = pd.read_csv(plant_capacity_path, dtype=str, keep_default_na=False)
if 'Plant' not in plant_capacity_df.columns or 'SupplyCapacity' not in plant_capacity_df.columns:
    raise KeyError("plant_capacity.csv must contain columns 'Plant' and 'SupplyCapacity'")
plant_capacity_df['Plant'] = plant_capacity_df['Plant'].str.strip()
plant_capacity_df['SupplyCapacity'] = plant_capacity_df['SupplyCapacity'].astype(float)
plants = plant_capacity_df['Plant'].tolist()
plant_capacities = dict(zip(plant_capacity_df['Plant'], plant_capacity_df['SupplyCapacity']))
retailer_demand_df = pd.read_csv(retailer_demand_path, dtype=str, keep_default_na=False)
if 'Retailer' not in retailer_demand_df.columns or 'Demand' not in retailer_demand_df.columns:
    raise KeyError("retailer_demand.csv must contain columns 'Retailer' and 'Demand'")
retailer_demand_df['Retailer'] = retailer_demand_df['Retailer'].str.strip()
retailer_demand_df['Demand'] = retailer_demand_df['Demand'].astype(float)
retailers = retailer_demand_df['Retailer'].tolist()
retailer_demands = dict(zip(retailer_demand_df['Retailer'], retailer_demand_df['Demand']))
route_var_costs_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
if 'Plant' not in route_var_costs_df.columns:
    raise KeyError("route_variable_costs.csv must contain column 'Plant'")
route_var_costs_df['Plant'] = route_var_costs_df['Plant'].str.strip()
for r in retailers:
    if r not in route_var_costs_df.columns:
        raise KeyError(f"route_variable_costs.csv missing retailer column '{r}'")
route_var_costs = {}
for (_, row) in route_var_costs_df.iterrows():
    i = row['Plant']
    for j in retailers:
        route_var_costs[i, j] = float(row[j])
route_fixed_costs_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
if 'Plant' not in route_fixed_costs_df.columns:
    raise KeyError("route_fixed_costs.csv must contain column 'Plant'")
route_fixed_costs_df['Plant'] = route_fixed_costs_df['Plant'].str.strip()
for r in retailers:
    if r not in route_fixed_costs_df.columns:
        raise KeyError(f"route_fixed_costs.csv missing retailer column '{r}'")
route_fixed_costs = {}
for (_, row) in route_fixed_costs_df.iterrows():
    i = row['Plant']
    for j in retailers:
        route_fixed_costs[i, j] = float(row[j])
M_ij = {}
for i in plants:
    for j in retailers:
        M_ij[i, j] = min(plant_capacities[i], retailer_demands[j])
for i in plants:
    for j in retailers:
        if (i, j) not in route_var_costs:
            raise KeyError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in route_fixed_costs:
            raise KeyError(f'Missing fixed cost for route ({i},{j})')
        if (i, j) not in M_ij:
            raise KeyError(f'Missing M_ij for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_costs[i, j] * x_vars[i, j] + route_fixed_costs[i, j] * y_vars[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == retailer_demands[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in retailers)) <= plant_capacities[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x_vars[i, j] <= M_ij[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()