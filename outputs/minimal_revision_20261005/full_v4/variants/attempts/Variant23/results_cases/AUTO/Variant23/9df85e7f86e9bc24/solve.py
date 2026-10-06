import gurobipy as gp
import pandas as pd
import numpy as np

def solve_fixed_charge_transportation():
    depot_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv'
    market_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv'
    route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv'
    route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv'
    depot_df = pd.read_csv(depot_capacity_path, sep=',')
    market_df = pd.read_csv(market_demand_path, sep=',')
    varcost_df = pd.read_csv(route_variable_costs_path, sep=',')
    fixedcost_df = pd.read_csv(route_fixed_costs_path, sep=',')
    depots = depot_df['Depot'].astype(str).str.strip().tolist()
    markets = market_df['Market'].astype(str).str.strip().tolist()
    S = {}
    for (_, row) in depot_df.iterrows():
        depot = str(row['Depot']).strip()
        S[depot] = int(row['SupplyCapacity'])
    D = {}
    for (_, row) in market_df.iterrows():
        market = str(row['Market']).strip()
        D[market] = int(row['Demand'])
    c = {}
    varcost_df['Depot'] = varcost_df['Depot'].astype(str).str.strip()
    for (_, row) in varcost_df.iterrows():
        depot = row['Depot']
        for market in markets:
            if market not in row:
                raise KeyError(f"Market '{market}' not found in route_variable_costs.csv columns")
            c[depot, market] = float(row[market])
    f = {}
    fixedcost_df['Depot'] = fixedcost_df['Depot'].astype(str).str.strip()
    for (_, row) in fixedcost_df.iterrows():
        depot = row['Depot']
        for market in markets:
            if market not in row:
                raise KeyError(f"Market '{market}' not found in route_fixed_costs.csv columns")
            f[depot, market] = float(row[market])
    M = {}
    for i in depots:
        for j in markets:
            if i not in S:
                raise KeyError(f"Depot '{i}' missing in depot_capacity.csv")
            if j not in D:
                raise KeyError(f"Market '{j}' missing in market_demand.csv")
            M[i, j] = min(S[i], D[j])
    for i in depots:
        for j in markets:
            if (i, j) not in c:
                raise KeyError(f'Missing variable cost for route ({i}, {j})')
            if (i, j) not in f:
                raise KeyError(f'Missing fixed cost for route ({i}, {j})')
    m = gp.Model('fixed_charge_transportation')
    m.setParam('MIPGap', 0.0001)
    route_keys = [(i, j) for i in depots for j in markets]
    x = m.addVars(route_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(route_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for (i, j) in route_keys)), gp.GRB.MINIMIZE)
    for j in markets:
        m.addConstr(gp.quicksum((x[i, j] for i in depots)) == D[j], name=f'demand_{j}')
    for i in depots:
        m.addConstr(gp.quicksum((x[i, j] for j in markets)) <= S[i], name=f'supply_{i}')
    for (i, j) in route_keys:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (i, j) in route_keys:
            print(f'{x[i, j].VarName} {x[i, j].X}')
            print(f'{y[i, j].VarName} {y[i, j].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_fixed_charge_transportation()