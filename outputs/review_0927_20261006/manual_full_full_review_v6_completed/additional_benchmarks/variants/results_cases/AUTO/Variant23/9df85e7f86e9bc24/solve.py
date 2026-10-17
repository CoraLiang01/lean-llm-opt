import gurobipy as gp
import pandas as pd
import numpy as np
depot_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv', dtype=str, keep_default_na=False)
market_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
depot_capacity_df['Depot'] = depot_capacity_df['Depot'].str.strip()
depots = depot_capacity_df['Depot'].tolist()
depot_capacities = depot_capacity_df.set_index('Depot')['SupplyCapacity'].astype(int).to_dict()
market_demand_df['Market'] = market_demand_df['Market'].str.strip()
markets = market_demand_df['Market'].tolist()
market_demands = market_demand_df.set_index('Market')['Demand'].astype(int).to_dict()
route_var_costs_df['Depot'] = route_var_costs_df['Depot'].str.strip()
route_fixed_costs_df['Depot'] = route_fixed_costs_df['Depot'].str.strip()
for d in depots:
    if d not in route_var_costs_df['Depot'].values:
        raise ValueError(f'Depot {d} missing in route_variable_costs.csv')
    if d not in route_fixed_costs_df['Depot'].values:
        raise ValueError(f'Depot {d} missing in route_fixed_costs.csv')
for m in markets:
    if m not in route_var_costs_df.columns:
        raise ValueError(f'Market {m} missing in route_variable_costs.csv columns')
    if m not in route_fixed_costs_df.columns:
        raise ValueError(f'Market {m} missing in route_fixed_costs.csv columns')
route_var_costs = {}
route_fixed_costs = {}
for d in depots:
    row_var = route_var_costs_df[route_var_costs_df['Depot'] == d].iloc[0]
    row_fixed = route_fixed_costs_df[route_fixed_costs_df['Depot'] == d].iloc[0]
    for m in markets:
        route_var_costs[d, m] = int(row_var[m])
        route_fixed_costs[d, m] = int(row_fixed[m])
M_ij = {}
for d in depots:
    S_i = depot_capacities[d]
    for m in markets:
        D_j = market_demands[m]
        M_ij[d, m] = min(S_i, D_j)
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(depots, markets, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(depots, markets, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_costs[d, m] * x_vars[d, m] + route_fixed_costs[d, m] * y_vars[d, m] for d in depots for m in markets)), gp.GRB.MINIMIZE)
for mkt in markets:
    m.addConstr(gp.quicksum((x_vars[d, mkt] for d in depots)) == market_demands[mkt], name=f'demand_{mkt}')
for dep in depots:
    m.addConstr(gp.quicksum((x_vars[dep, m] for m in markets)) <= depot_capacities[dep], name=f'supply_{dep}')
for d in depots:
    for m in markets:
        m.addConstr(x_vars[d, m] <= M_ij[d, m] * y_vars[d, m], name=f'link_{d}_{m}')
m.optimize()