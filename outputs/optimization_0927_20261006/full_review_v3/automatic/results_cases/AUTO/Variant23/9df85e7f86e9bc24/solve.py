import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(x):
    return str(x).strip()
depot_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv', dtype=str, keep_default_na=False)
market_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
depot_ids = depot_capacity_df['Depot'].apply(norm_str).tolist()
depot_supply = {norm_str(row['Depot']): int(row['SupplyCapacity']) for (_, row) in depot_capacity_df.iterrows()}
market_ids = market_demand_df['Market'].apply(norm_str).tolist()
market_demand = {norm_str(row['Market']): int(row['Demand']) for (_, row) in market_demand_df.iterrows()}
route_var_costs = {}
for (_, row) in route_var_costs_df.iterrows():
    depot = norm_str(row['Depot'])
    for market in market_ids:
        route_var_costs[depot, market] = int(row[market])
route_fixed_costs = {}
for (_, row) in route_fixed_costs_df.iterrows():
    depot = norm_str(row['Depot'])
    for market in market_ids:
        route_fixed_costs[depot, market] = int(row[market])
M_ij = {}
for depot in depot_ids:
    for market in market_ids:
        M_ij[depot, market] = min(depot_supply[depot], market_demand[market])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars([(i, j) for i in depot_ids for j in market_ids], vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars([(i, j) for i in depot_ids for j in market_ids], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_costs[i, j] * x_vars[i, j] + route_fixed_costs[i, j] * y_vars[i, j] for i in depot_ids for j in market_ids)), gp.GRB.MINIMIZE)
for j in market_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in depot_ids)) == market_demand[j], name='')
for i in depot_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in market_ids)) <= depot_supply[i], name='')
for i in depot_ids:
    for j in market_ids:
        m.addConstr(x_vars[i, j] <= M_ij[i, j] * y_vars[i, j], name='')
m.optimize()