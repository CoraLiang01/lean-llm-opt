import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'

def solve_transportation():
    cost_df = pd.read_csv(cost_matrix_path, sep=',')
    sources_df = pd.read_csv(sources_path, sep=',')
    dest_df = pd.read_csv(destinations_path, sep=',')
    sources = [f'S{i}' for i in range(1, 11)]
    destinations = [f'D{j}' for j in range(1, 21)]
    missing_sources = set(sources) - set(sources_df['source_id'].astype(str))
    if missing_sources:
        raise ValueError(f'Missing sources in sources file: {missing_sources}')
    missing_dest = set(destinations) - set(dest_df['destination_id'].astype(str))
    if missing_dest:
        raise ValueError(f'Missing destinations in destinations file: {missing_dest}')
    missing_cost_sources = set(sources) - set(cost_df['source_id'].astype(str))
    if missing_cost_sources:
        raise ValueError(f'Missing sources in cost matrix: {missing_cost_sources}')
    missing_cost_dest = set(destinations) - set(cost_df.columns[1:])
    if missing_cost_dest:
        raise ValueError(f'Missing destinations in cost matrix columns: {missing_cost_dest}')
    supply = dict(zip(sources_df['source_id'].astype(str), sources_df['supply_units']))
    demand = dict(zip(dest_df['destination_id'].astype(str), dest_df['demand_units']))
    cost = {}
    cost_df_indexed = cost_df.set_index('source_id')
    for i in sources:
        for j in destinations:
            cost[i, j] = float(cost_df_indexed.loc[i, j])
    m = gp.Model('TruckTransportation')
    x = m.addVars(sources, destinations, name='x', lb=0.0, vtype=gp.GRB.CONTINUOUS)
    t = m.addVars(sources, destinations, name='t', lb=0, vtype=gp.GRB.INTEGER)
    m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in sources for j in destinations)), gp.GRB.MINIMIZE)
    for i in sources:
        m.addConstr(gp.quicksum((x[i, j] for j in destinations)) <= supply[i], name=f'supply_{i}')
    for j in destinations:
        m.addConstr(gp.quicksum((x[i, j] for i in sources)) == demand[j], name=f'demand_{j}')
    for i in sources:
        for j in destinations:
            m.addConstr(x[i, j] <= 10 * t[i, j], name=f'truckcap_{i}_{j}')
    m.optimize()
    return m
m = solve_transportation()