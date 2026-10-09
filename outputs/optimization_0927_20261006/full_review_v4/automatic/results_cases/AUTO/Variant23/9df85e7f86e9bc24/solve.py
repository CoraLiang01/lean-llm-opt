import gurobipy as gp
import pandas as pd
import numpy as np
import re
depot_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv'
market_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv'
depot_df = pd.read_csv(depot_capacity_path, dtype=str, keep_default_na=False)
market_df = pd.read_csv(market_demand_path, dtype=str, keep_default_na=False)
varcost_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
fixedcost_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
depot_df['Depot'] = depot_df['Depot'].str.strip()
market_df['Market'] = market_df['Market'].str.strip()
varcost_df['Depot'] = varcost_df['Depot'].str.strip()
fixedcost_df['Depot'] = fixedcost_df['Depot'].str.strip()
depots = depot_df['Depot'].tolist()
markets = market_df['Market'].tolist()
depot_capacity = {}
for (_, row) in depot_df.iterrows():
    depot_id = row['Depot']
    try:
        depot_capacity[depot_id] = int(row['SupplyCapacity'])
    except Exception as e:
        raise ValueError(f"Invalid SupplyCapacity for depot {depot_id}: {row['SupplyCapacity']}") from e
market_demand = {}
for (_, row) in market_df.iterrows():
    market_id = row['Market']
    try:
        market_demand[market_id] = int(row['Demand'])
    except Exception as e:
        raise ValueError(f"Invalid Demand for market {market_id}: {row['Demand']}") from e
route_var_cost = {}
for (_, row) in varcost_df.iterrows():
    depot_id = row['Depot']
    for market_id in markets:
        try:
            route_var_cost[depot_id, market_id] = float(row[market_id])
        except Exception as e:
            raise ValueError(f'Invalid variable cost for route ({depot_id}, {market_id}): {row[market_id]}') from e
route_fixed_cost = {}
for (_, row) in fixedcost_df.iterrows():
    depot_id = row['Depot']
    for market_id in markets:
        try:
            route_fixed_cost[depot_id, market_id] = float(row[market_id])
        except Exception as e:
            raise ValueError(f'Invalid fixed cost for route ({depot_id}, {market_id}): {row[market_id]}') from e
route_link_ub = {}
for i in depots:
    for j in markets:
        if i not in depot_capacity:
            raise KeyError(f'Depot {i} missing from depot_capacity')
        if j not in market_demand:
            raise KeyError(f'Market {j} missing from market_demand')
        route_link_ub[i, j] = min(depot_capacity[i], market_demand[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(depots, markets, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(depots, markets, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_cost[i, j] * x_vars[i, j] + route_fixed_cost[i, j] * y_vars[i, j] for i in depots for j in markets)), gp.GRB.MINIMIZE)
for j in markets:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in depots)) == market_demand[j], name=f'demand_{j}')
for i in depots:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in markets)) <= depot_capacity[i], name=f'supply_{i}')
for i in depots:
    for j in markets:
        m.addConstr(x_vars[i, j] <= route_link_ub[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()