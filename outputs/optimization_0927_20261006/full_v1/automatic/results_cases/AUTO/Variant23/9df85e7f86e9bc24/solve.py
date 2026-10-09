import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
depot_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv', dtype=str, keep_default_na=False)
market_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
depots = depot_capacity_df['Depot'].apply(lambda x: x.strip()).tolist()
markets = market_demand_df['Market'].apply(lambda x: x.strip()).tolist()
depot_capacity_dict = {}
for (_, row) in depot_capacity_df.iterrows():
    depot = row['Depot'].strip()
    try:
        cap = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f"Invalid SupplyCapacity for depot {depot}: {row['SupplyCapacity']}")
    depot_capacity_dict[depot] = cap
market_demand_dict = {}
for (_, row) in market_demand_df.iterrows():
    market = row['Market'].strip()
    try:
        demand = int(row['Demand'])
    except Exception:
        raise ValueError(f"Invalid Demand for market {market}: {row['Demand']}")
    market_demand_dict[market] = demand
route_var_costs_dict = {}
for (_, row) in route_var_costs_df.iterrows():
    depot = row['Depot'].strip()
    for market in markets:
        try:
            cost = float(row[market])
        except Exception:
            raise ValueError(f'Invalid variable cost for route ({depot},{market}): {row[market]}')
        route_var_costs_dict[depot, market] = cost
route_fixed_costs_dict = {}
for (_, row) in route_fixed_costs_df.iterrows():
    depot = row['Depot'].strip()
    for market in markets:
        try:
            cost = float(row[market])
        except Exception:
            raise ValueError(f'Invalid fixed cost for route ({depot},{market}): {row[market]}')
        route_fixed_costs_dict[depot, market] = cost
M_dict = {}
for i in depots:
    for j in markets:
        S_i = depot_capacity_dict[i]
        D_j = market_demand_dict[j]
        M_dict[i, j] = min(S_i, D_j)
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars([(i, j) for i in depots for j in markets], vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars([(i, j) for i in depots for j in markets], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_costs_dict[i, j] * x_vars[i, j] + route_fixed_costs_dict[i, j] * y_vars[i, j] for i in depots for j in markets)), gp.GRB.MINIMIZE)
for j in markets:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in depots)) == market_demand_dict[j], name='')
for i in depots:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in markets)) <= depot_capacity_dict[i], name='')
for i in depots:
    for j in markets:
        m.addConstr(x_vars[i, j] <= M_dict[i, j] * y_vars[i, j], name='')
m.optimize()