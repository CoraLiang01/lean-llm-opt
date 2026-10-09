import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',', dtype=str, keep_default_na=False)
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
sources_df = pd.read_csv(sources_path, sep=',', dtype=str, keep_default_na=False)
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
dest_df = pd.read_csv(destinations_path, sep=',', dtype=str, keep_default_na=False)
sources_df['source_id'] = sources_df['source_id'].str.strip()
cost_df['source_id'] = cost_df['source_id'].str.strip()
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
missing_sources = set(sources) - set(sources_df['source_id'])
if missing_sources:
    raise ValueError(f'Missing sources in expanded_sources.csv: {missing_sources}')
missing_sources_cost = set(sources) - set(cost_df['source_id'])
if missing_sources_cost:
    raise ValueError(f'Missing sources in expanded_cost_matrix.csv: {missing_sources_cost}')
missing_destinations = set(destinations) - set(dest_df['destination_id'])
if missing_destinations:
    raise ValueError(f'Missing destinations in expanded_destinations.csv: {missing_destinations}')
missing_destinations_cost = set(destinations) - set(cost_df.columns[1:])
if missing_destinations_cost:
    raise ValueError(f'Missing destinations in expanded_cost_matrix.csv columns: {missing_destinations_cost}')
cost_dict = {}
for (_, row) in cost_df.iterrows():
    s = row['source_id']
    for d in destinations:
        try:
            cost_dict[s, d] = float(row[d])
        except Exception as e:
            raise ValueError(f'Invalid or missing cost for source {s}, destination {d}: {e}')
supply_dict = {}
for (_, row) in sources_df.iterrows():
    s = row['source_id']
    try:
        supply_dict[s] = int(row['supply_units'])
    except Exception as e:
        raise ValueError(f'Invalid or missing supply_units for source {s}: {e}')
demand_dict = {}
for (_, row) in dest_df.iterrows():
    d = row['destination_id']
    try:
        demand_dict[d] = int(row['demand_units'])
    except Exception as e:
        raise ValueError(f'Invalid or missing demand_units for destination {d}: {e}')
TRUCK_CAPACITY = 10
m = gp.Model('Transportation_MILP')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply_dict[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand_dict[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(x_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'truck_load_{s}_{d}')
m.setObjective(gp.quicksum((cost_dict[s, d] * x_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
m.optimize()