import gurobipy as gp
import pandas as pd
import numpy as np
depot_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv', sep=',')
market_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv', sep=',')
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv', sep=',')
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv', sep=',')
depots = depot_capacity_df['Depot'].astype(str).str.strip().tolist()
markets = market_demand_df['Market'].astype(str).str.strip().tolist()
S = dict(zip(depot_capacity_df['Depot'].astype(str).str.strip(), depot_capacity_df['SupplyCapacity']))
D = dict(zip(market_demand_df['Market'].astype(str).str.strip(), market_demand_df['Demand']))
route_var_costs_df['Depot'] = route_var_costs_df['Depot'].astype(str).str.strip()
c = {}
for i in depots:
    row = route_var_costs_df.loc[route_var_costs_df['Depot'] == i]
    if row.empty:
        raise ValueError(f'Depot {i} not found in route_variable_costs.csv')
    for j in markets:
        if j not in row.columns:
            raise ValueError(f'Market {j} not found in route_variable_costs.csv columns')
        c[i, j] = float(row.iloc[0][j])
route_fixed_costs_df['Depot'] = route_fixed_costs_df['Depot'].astype(str).str.strip()
f = {}
for i in depots:
    row = route_fixed_costs_df.loc[route_fixed_costs_df['Depot'] == i]
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
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Route Plan ---')
    for i in depots:
        for j in markets:
            xval = x[i, j].X
            yval = y[i, j].X
            if yval > 0.5:
                print(f'Route {i} -> {j}: Activated (y=1), Shipment={xval:.2f}, VarCost={c[i, j]}, FixedCost={f[i, j]}')
            elif xval > 1e-06:
                print(f'WARNING: x[{i},{j}]={xval:.2f} > 0 but y[{i},{j}]=0')
else:
    print(f'No optimal solution found. Status: {m.status}')