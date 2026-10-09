import gurobipy as gp
import pandas as pd
import numpy as np
depot_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv', dtype=str, keep_default_na=False)
market_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
depot_capacity_df['Depot'] = depot_capacity_df['Depot'].str.strip()
market_demand_df['Market'] = market_demand_df['Market'].str.strip()
route_var_costs_df['Depot'] = route_var_costs_df['Depot'].str.strip()
route_fixed_costs_df['Depot'] = route_fixed_costs_df['Depot'].str.strip()
depots = depot_capacity_df['Depot'].unique().tolist()
markets = market_demand_df['Market'].unique().tolist()
S = {}
for (_, row) in depot_capacity_df.iterrows():
    depot = row['Depot']
    try:
        S[depot] = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f"Invalid SupplyCapacity for depot {depot}: {row['SupplyCapacity']}")
D = {}
for (_, row) in market_demand_df.iterrows():
    market = row['Market']
    try:
        D[market] = int(row['Demand'])
    except Exception:
        raise ValueError(f"Invalid Demand for market {market}: {row['Demand']}")
c = {}
for (_, row) in route_var_costs_df.iterrows():
    depot = row['Depot']
    for market in markets:
        try:
            c[depot, market] = float(row[market])
        except Exception:
            raise ValueError(f'Missing or invalid variable cost for route ({depot}, {market})')
f = {}
for (_, row) in route_fixed_costs_df.iterrows():
    depot = row['Depot']
    for market in markets:
        try:
            f[depot, market] = float(row[market])
        except Exception:
            raise ValueError(f'Missing or invalid fixed cost for route ({depot}, {market})')
M = {}
for i in depots:
    for j in markets:
        if i not in S:
            raise ValueError(f'Depot {i} missing in S')
        if j not in D:
            raise ValueError(f'Market {j} missing in D')
        M[i, j] = min(S[i], D[j])
route_keys = [(i, j) for i in depots for j in markets]

def solve_fixed_charge_transportation():
    m = gp.Model('fixed_charge_transportation')
    x_vars = m.addVars(route_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(route_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for (i, j) in route_keys)), gp.GRB.MINIMIZE)
    for j in markets:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in depots)) == D[j], name=f'demand_{j}')
    for i in depots:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in markets)) <= S[i], name=f'supply_{i}')
    for (i, j) in route_keys:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_fixed_charge_transportation()