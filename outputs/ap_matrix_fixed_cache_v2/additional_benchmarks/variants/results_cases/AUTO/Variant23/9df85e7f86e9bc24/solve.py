import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip()
depot_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv', dtype=str, keep_default_na=False)
market_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
depot_ids = [norm_str(d) for d in depot_capacity_df['Depot']]
market_ids = [norm_str(m) for m in market_demand_df['Market']]
depot_capacity_df['Depot'] = depot_capacity_df['Depot'].apply(norm_str)
depot_capacity_df['SupplyCapacity'] = depot_capacity_df['SupplyCapacity'].astype(int)
S = dict(zip(depot_capacity_df['Depot'], depot_capacity_df['SupplyCapacity']))
market_demand_df['Market'] = market_demand_df['Market'].apply(norm_str)
market_demand_df['Demand'] = market_demand_df['Demand'].astype(int)
D = dict(zip(market_demand_df['Market'], market_demand_df['Demand']))
route_var_costs_df['Depot'] = route_var_costs_df['Depot'].apply(norm_str)
c = {}
for _, row in route_var_costs_df.iterrows():
    i = row['Depot']
    for j in market_ids:
        c[i, j] = int(row[j])
route_fixed_costs_df['Depot'] = route_fixed_costs_df['Depot'].apply(norm_str)
f = {}
for _, row in route_fixed_costs_df.iterrows():
    i = row['Depot']
    for j in market_ids:
        f[i, j] = int(row[j])
M = {}
for i in depot_ids:
    for j in market_ids:
        M[i, j] = min(S[i], D[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars([(i, j) for i in depot_ids for j in market_ids], vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars([(i, j) for i in depot_ids for j in market_ids], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in depot_ids for j in market_ids)), gp.GRB.MINIMIZE)
for j in market_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in depot_ids)) == D[j], name=f'demand_{j}')
for i in depot_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in market_ids)) <= S[i], name=f'supply_{i}')
for i in depot_ids:
    for j in market_ids:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()