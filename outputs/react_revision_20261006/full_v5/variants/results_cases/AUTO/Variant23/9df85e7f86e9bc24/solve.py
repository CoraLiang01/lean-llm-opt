import gurobipy as gp
import pandas as pd
import numpy as np
depot_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv'
market_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv'
depot_df = pd.read_csv(depot_capacity_path, sep=',')
market_df = pd.read_csv(market_demand_path, sep=',')
varcost_df = pd.read_csv(route_variable_costs_path, sep=',')
fixedcost_df = pd.read_csv(route_fixed_costs_path, sep=',')
depot_df['Depot'] = depot_df['Depot'].astype(str).str.strip()
market_df['Market'] = market_df['Market'].astype(str).str.strip()
varcost_df['Depot'] = varcost_df['Depot'].astype(str).str.strip()
fixedcost_df['Depot'] = fixedcost_df['Depot'].astype(str).str.strip()
depots = depot_df['Depot'].unique().tolist()
markets = market_df['Market'].unique().tolist()
for d in depots:
    if d not in varcost_df['Depot'].values:
        raise ValueError(f'Depot {d} missing in route_variable_costs.csv')
    if d not in fixedcost_df['Depot'].values:
        raise ValueError(f'Depot {d} missing in route_fixed_costs.csv')
for m in markets:
    if m not in varcost_df.columns:
        raise ValueError(f'Market {m} missing as column in route_variable_costs.csv')
    if m not in fixedcost_df.columns:
        raise ValueError(f'Market {m} missing as column in route_fixed_costs.csv')
S = depot_df.set_index('Depot')['SupplyCapacity'].to_dict()
D = market_df.set_index('Market')['Demand'].to_dict()
c = {}
f = {}
M = {}
for i in depots:
    for j in markets:
        c_ij = varcost_df.loc[varcost_df['Depot'] == i, j]
        if c_ij.empty:
            raise ValueError(f'Missing variable cost for route ({i},{j})')
        c[i, j] = float(c_ij.values[0])
        f_ij = fixedcost_df.loc[fixedcost_df['Depot'] == i, j]
        if f_ij.empty:
            raise ValueError(f'Missing fixed cost for route ({i},{j})')
        f[i, j] = float(f_ij.values[0])
        M[i, j] = float(min(S[i], D[j]))
route_keys = [(i, j) for i in depots for j in markets]
m = gp.Model('FixedChargeTransportation')
x = m.addVars(route_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(route_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for (i, j) in route_keys)), gp.GRB.MINIMIZE)
for j in markets:
    m.addConstr(gp.quicksum((x[i, j] for i in depots)) == D[j], name=f'demand_{j}')
for i in depots:
    m.addConstr(gp.quicksum((x[i, j] for j in markets)) <= S[i], name=f'supply_{i}')
for (i, j) in route_keys:
    m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for (i, j) in route_keys:
        print(f'{x[i, j].VarName} {x[i, j].X}')
        print(f'{y[i, j].VarName} {y[i, j].X}')
else:
    print(f'Solver status: {m.Status}')