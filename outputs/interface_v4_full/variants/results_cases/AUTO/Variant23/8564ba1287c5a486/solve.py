import gurobipy as gp
import pandas as pd
import numpy as np
depot_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/depot_capacity.csv'
market_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/market_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/route_fixed_costs.csv'
depot_df = pd.read_csv(depot_capacity_path, sep=',')
market_df = pd.read_csv(market_demand_path, sep=',')
var_costs_df = pd.read_csv(route_variable_costs_path, sep=',')
fixed_costs_df = pd.read_csv(route_fixed_costs_path, sep=',')
depot_df['Depot'] = depot_df['Depot'].astype(str).str.strip()
market_df['Market'] = market_df['Market'].astype(str).str.strip()
var_costs_df['Depot'] = var_costs_df['Depot'].astype(str).str.strip()
fixed_costs_df['Depot'] = fixed_costs_df['Depot'].astype(str).str.strip()
depots = list(depot_df['Depot'])
markets = list(market_df['Market'])
S = depot_df.set_index('Depot')['SupplyCapacity'].to_dict()
D = market_df.set_index('Market')['Demand'].to_dict()
c = {}
for i in depots:
    row = var_costs_df[var_costs_df['Depot'] == i]
    if row.empty:
        raise ValueError(f'Depot {i} not found in route_variable_costs.csv')
    for j in markets:
        if j not in row.columns:
            raise ValueError(f'Market {j} not found in route_variable_costs.csv columns')
        c[i, j] = float(row.iloc[0][j])
f = {}
for i in depots:
    row = fixed_costs_df[fixed_costs_df['Depot'] == i]
    if row.empty:
        raise ValueError(f'Depot {i} not found in route_fixed_costs.csv')
    for j in markets:
        if j not in row.columns:
            raise ValueError(f'Market {j} not found in route_fixed_costs.csv columns')
        f[i, j] = float(row.iloc[0][j])
M = {}
for i in depots:
    for j in markets:
        M[i, j] = min(S[i], D[j])
for i in depots:
    if i not in S:
        raise ValueError(f'Depot {i} missing in depot_capacity.csv')
for j in markets:
    if j not in D:
        raise ValueError(f'Market {j} missing in market_demand.csv')
for i in depots:
    for j in markets:
        if (i, j) not in c:
            raise ValueError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in f:
            raise ValueError(f'Missing fixed cost for route ({i},{j})')
        if (i, j) not in M:
            raise ValueError(f'Missing Big-M for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x = m.addVars(depots, markets, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y = m.addVars(depots, markets, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for i in depots for j in markets)), gp.GRB.MINIMIZE)
for j in markets:
    m.addConstr(gp.quicksum((x[i, j] for i in depots)) == D[j], name=f'demand_{j}')
for i in depots:
    m.addConstr(gp.quicksum((x[i, j] for j in markets)) <= S[i], name=f'supply_{i}')
for i in depots:
    for j in markets:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
m.optimize()