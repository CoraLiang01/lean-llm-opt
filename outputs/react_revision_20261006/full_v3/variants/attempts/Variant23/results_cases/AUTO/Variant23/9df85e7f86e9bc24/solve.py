import gurobipy as gp
import pandas as pd
import numpy as np
depot_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/depot_capacity.csv'
market_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/market_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant23/inputs/route_fixed_costs.csv'

def solve_fixed_charge_transportation():
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
    S = depot_df.set_index('Depot')['SupplyCapacity'].to_dict()
    D = market_df.set_index('Market')['Demand'].to_dict()
    c = {}
    for (_, row) in varcost_df.iterrows():
        i = str(row['Depot']).strip()
        for j in markets:
            c[i, j] = float(row[j])
    f = {}
    for (_, row) in fixedcost_df.iterrows():
        i = str(row['Depot']).strip()
        for j in markets:
            f[i, j] = float(row[j])
    M = {}
    for i in depots:
        for j in markets:
            if i not in S:
                raise ValueError(f'Depot {i} missing in depot capacities.')
            if j not in D:
                raise ValueError(f'Market {j} missing in market demands.')
            M[i, j] = min(S[i], D[j])
    for i in depots:
        for j in markets:
            if (i, j) not in c:
                raise ValueError(f'Missing variable cost for route ({i}, {j})')
            if (i, j) not in f:
                raise ValueError(f'Missing fixed cost for route ({i}, {j})')
    route_keys = [(i, j) for i in depots for j in markets]
    m = gp.Model('fixed_charge_transportation')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(route_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(route_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for (i, j) in route_keys)), gp.GRB.MINIMIZE)
    for j in markets:
        m.addConstr(gp.quicksum((x[i, j] for i in depots)) == D[j], name='demand')
    for i in depots:
        m.addConstr(gp.quicksum((x[i, j] for j in markets)) <= S[i], name='supply')
    for (i, j) in route_keys:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name='link')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for (i, j) in route_keys:
            print(f'{x[i, j].VarName} {x[i, j].X}')
            print(f'{y[i, j].VarName} {y[i, j].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_fixed_charge_transportation()