import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',')
sources_df = pd.read_csv(sources_path, sep=',')
dest_df = pd.read_csv(destinations_path, sep=',')
cost_df['source_id'] = cost_df['source_id'].astype(str).str.strip()
sources_df['source_id'] = sources_df['source_id'].astype(str).str.strip()
dest_df['destination_id'] = dest_df['destination_id'].astype(str).str.strip()
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
supply_dict = {}
for s in sources:
    row = sources_df[sources_df['source_id'] == s]
    if row.empty:
        raise ValueError(f'Missing supply data for source {s}')
    supply_dict[s] = int(row.iloc[0]['supply_units'])
demand_dict = {}
for d in destinations:
    row = dest_df[dest_df['destination_id'] == d]
    if row.empty:
        raise ValueError(f'Missing demand data for destination {d}')
    demand_dict[d] = int(row.iloc[0]['demand_units'])
cost_dict = {}
for s in sources:
    row = cost_df[cost_df['source_id'] == s]
    if row.empty:
        raise ValueError(f'Missing cost data for source {s}')
    for d in destinations:
        if d not in cost_df.columns:
            raise ValueError(f'Missing cost column for destination {d}')
        cost = float(row.iloc[0][d])
        cost_dict[s, d] = cost
truck_capacity = 10
m = gp.Model('transportation_milp')
trucks = m.addVars(sources, destinations, vtype=GRB.INTEGER, lb=0, name='')
cargo = m.addVars(sources, destinations, vtype=GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_dict[s, d] * cargo[s, d] for s in sources for d in destinations)), GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((cargo[s, d] for d in destinations)) <= supply_dict[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((cargo[s, d] for s in sources)) == demand_dict[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(cargo[s, d] <= truck_capacity * trucks[s, d], name=f'truckcap_{s}_{d}')
m.optimize()