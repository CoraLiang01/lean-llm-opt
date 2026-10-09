import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(x):
    return str(x).strip().casefold()
depot_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv', dtype=str, keep_default_na=False)
market_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
depot_capacity_df['Depot_norm'] = depot_capacity_df['Depot'].apply(norm_str)
market_demand_df['Market_norm'] = market_demand_df['Market'].apply(norm_str)
route_var_costs_df['Depot_norm'] = route_var_costs_df['Depot'].apply(norm_str)
route_fixed_costs_df['Depot_norm'] = route_fixed_costs_df['Depot'].apply(norm_str)
depots = list(depot_capacity_df['Depot'])
depots_norm = [norm_str(d) for d in depots]
markets = list(market_demand_df['Market'])
markets_norm = [norm_str(m) for m in markets]
depot_capacity = {}
for (idx, row) in depot_capacity_df.iterrows():
    depot = row['Depot']
    depot_norm = row['Depot_norm']
    try:
        cap = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f'Invalid SupplyCapacity for depot {depot}')
    depot_capacity[depot] = cap
market_demand = {}
for (idx, row) in market_demand_df.iterrows():
    market = row['Market']
    market_norm = row['Market_norm']
    try:
        dem = int(row['Demand'])
    except Exception:
        raise ValueError(f'Invalid Demand for market {market}')
    market_demand[market] = dem
route_var_cost = {}
for (idx, row) in route_var_costs_df.iterrows():
    depot = row['Depot']
    depot_norm = row['Depot_norm']
    for (market, market_norm) in zip(markets, markets_norm):
        try:
            cost = float(row[market])
        except Exception:
            raise ValueError(f'Missing or invalid variable cost for route ({depot}, {market})')
        route_var_cost[depot, market] = cost
route_fixed_cost = {}
for (idx, row) in route_fixed_costs_df.iterrows():
    depot = row['Depot']
    depot_norm = row['Depot_norm']
    for (market, market_norm) in zip(markets, markets_norm):
        try:
            cost = float(row[market])
        except Exception:
            raise ValueError(f'Missing or invalid fixed cost for route ({depot}, {market})')
        route_fixed_cost[depot, market] = cost
M_ij = {}
for depot in depots:
    for market in markets:
        cap = depot_capacity[depot]
        dem = market_demand[market]
        M_ij[depot, market] = min(cap, dem)
for depot in depots:
    if depot not in depot_capacity:
        raise KeyError(f'Depot {depot} missing in depot_capacity')
for market in markets:
    if market not in market_demand:
        raise KeyError(f'Market {market} missing in market_demand')
for depot in depots:
    for market in markets:
        if (depot, market) not in route_var_cost:
            raise KeyError(f'Route variable cost missing for ({depot}, {market})')
        if (depot, market) not in route_fixed_cost:
            raise KeyError(f'Route fixed cost missing for ({depot}, {market})')
        if (depot, market) not in M_ij:
            raise KeyError(f'M_ij missing for ({depot}, {market})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(depots, markets, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(depots, markets, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_cost[i, j] * x_vars[i, j] + route_fixed_cost[i, j] * y_vars[i, j] for i in depots for j in markets)), gp.GRB.MINIMIZE)
for j in markets:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in depots)) == market_demand[j])
for i in depots:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in markets)) <= depot_capacity[i])
for i in depots:
    for j in markets:
        m.addConstr(x_vars[i, j] <= M_ij[i, j] * y_vars[i, j])
m.optimize()