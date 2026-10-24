import gurobipy as gp
import pandas as pd
import numpy as np
depot_capacity_path = 'benchmark_dataset/Variant23/inputs/depot_capacity.csv'
depot_capacity_df = pd.read_csv(depot_capacity_path, dtype=str, keep_default_na=False)
if 'Depot' not in depot_capacity_df.columns or 'SupplyCapacity' not in depot_capacity_df.columns:
    raise KeyError("depot_capacity.csv must contain columns 'Depot' and 'SupplyCapacity'")
depot_capacity_df['Depot'] = depot_capacity_df['Depot'].str.strip()
depot_capacity_df['SupplyCapacity'] = depot_capacity_df['SupplyCapacity'].astype(float)
depots = depot_capacity_df['Depot'].tolist()
S = dict(zip(depot_capacity_df['Depot'], depot_capacity_df['SupplyCapacity']))
market_demand_path = 'benchmark_dataset/Variant23/inputs/market_demand.csv'
market_demand_df = pd.read_csv(market_demand_path, dtype=str, keep_default_na=False)
if 'Market' not in market_demand_df.columns or 'Demand' not in market_demand_df.columns:
    raise KeyError("market_demand.csv must contain columns 'Market' and 'Demand'")
market_demand_df['Market'] = market_demand_df['Market'].str.strip()
market_demand_df['Demand'] = market_demand_df['Demand'].astype(float)
markets = market_demand_df['Market'].tolist()
D = dict(zip(market_demand_df['Market'], market_demand_df['Demand']))
route_variable_costs_path = 'benchmark_dataset/Variant23/inputs/route_variable_costs.csv'
route_variable_costs_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
if 'Depot' not in route_variable_costs_df.columns:
    raise KeyError("route_variable_costs.csv must contain column 'Depot'")
route_variable_costs_df['Depot'] = route_variable_costs_df['Depot'].str.strip()
for m in markets:
    if m not in route_variable_costs_df.columns:
        raise KeyError(f"route_variable_costs.csv missing market column '{m}'")
route_variable_costs_df[markets] = route_variable_costs_df[markets].astype(float)
c = {}
for (_, row) in route_variable_costs_df.iterrows():
    i = row['Depot']
    for j in markets:
        c[i, j] = float(row[j])
route_fixed_costs_path = 'benchmark_dataset/Variant23/inputs/route_fixed_costs.csv'
route_fixed_costs_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
if 'Depot' not in route_fixed_costs_df.columns:
    raise KeyError("route_fixed_costs.csv must contain column 'Depot'")
route_fixed_costs_df['Depot'] = route_fixed_costs_df['Depot'].str.strip()
for m in markets:
    if m not in route_fixed_costs_df.columns:
        raise KeyError(f"route_fixed_costs.csv missing market column '{m}'")
route_fixed_costs_df[markets] = route_fixed_costs_df[markets].astype(float)
f = {}
for (_, row) in route_fixed_costs_df.iterrows():
    i = row['Depot']
    for j in markets:
        f[i, j] = float(row[j])
routes = [(i, j) for i in depots for j in markets]
M = {}
for i in depots:
    if i not in S:
        raise KeyError(f"Depot '{i}' missing from depot_capacity.csv")
    for j in markets:
        if j not in D:
            raise KeyError(f"Market '{j}' missing from market_demand.csv")
        M[i, j] = min(S[i], D[j])
for (i, j) in routes:
    if (i, j) not in c:
        raise KeyError(f'Missing variable cost for route ({i},{j})')
    if (i, j) not in f:
        raise KeyError(f'Missing fixed cost for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(routes, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(routes, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for (i, j) in routes)), gp.GRB.MINIMIZE)
for j in markets:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in depots)) >= D[j])
for i in depots:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in markets)) <= S[i])
for (i, j) in routes:
    m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j])
m.optimize()