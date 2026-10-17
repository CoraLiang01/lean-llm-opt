import gurobipy as gp
import pandas as pd
import numpy as np
import re
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_df = pd.read_csv(sources_path, sep=',', dtype=str, keep_default_na=False)
destinations_df = pd.read_csv(destinations_path, sep=',', dtype=str, keep_default_na=False)
cost_matrix_df = pd.read_csv(cost_matrix_path, sep=',', dtype=str, keep_default_na=False)
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
if 'source_id' not in sources_df.columns:
    raise KeyError("expanded_sources.csv must contain a 'source_id' column.")
if 'supply_units' not in sources_df.columns:
    raise KeyError("expanded_sources.csv must contain a 'supply_units' column.")
if 'destination_id' not in destinations_df.columns:
    raise KeyError("expanded_destinations.csv must contain a 'destination_id' column.")
if 'demand_units' not in destinations_df.columns:
    raise KeyError("expanded_destinations.csv must contain a 'demand_units' column.")
sources_df = sources_df[sources_df['source_id'].isin(sources)].copy()
destinations_df = destinations_df[destinations_df['destination_id'].isin(destinations)].copy()
if set(sources) != set(sources_df['source_id']):
    missing = set(sources) - set(sources_df['source_id'])
    raise ValueError(f'Missing source(s) in expanded_sources.csv: {missing}')
if set(destinations) != set(destinations_df['destination_id']):
    missing = set(destinations) - set(destinations_df['destination_id'])
    raise ValueError(f'Missing destination(s) in expanded_destinations.csv: {missing}')
supply_units = {}
for idx, row in sources_df.iterrows():
    try:
        supply_units[row['source_id']] = float(row['supply_units'])
    except Exception:
        raise ValueError(f"Invalid supply_units for source {row['source_id']}: {row['supply_units']}")
demand_units = {}
for idx, row in destinations_df.iterrows():
    try:
        demand_units[row['destination_id']] = float(row['demand_units'])
    except Exception:
        raise ValueError(f"Invalid demand_units for destination {row['destination_id']}: {row['demand_units']}")
cost_matrix_df = cost_matrix_df.set_index(cost_matrix_df.columns[0])
cost_matrix_df.index = cost_matrix_df.index.astype(str)
cost_matrix_df.columns = [str(c) for c in cost_matrix_df.columns]
if not set(sources).issubset(set(cost_matrix_df.index)):
    missing = set(sources) - set(cost_matrix_df.index)
    raise ValueError(f'Missing source(s) in expanded_cost_matrix.csv: {missing}')
if not set(destinations).issubset(set(cost_matrix_df.columns)):
    missing = set(destinations) - set(cost_matrix_df.columns)
    raise ValueError(f'Missing destination(s) in expanded_cost_matrix.csv: {missing}')
cost = {}
for s in sources:
    for d in destinations:
        val = cost_matrix_df.loc[s, d]
        try:
            cost[s, d] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for ({s},{d}): {val}')
m = gp.Model('IntegerTruckTransportation')
truck_capacity = 10.0
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply_units[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand_units[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(x_vars[s, d] <= truck_capacity * t_vars[s, d], name=f'truckload_{s}_{d}')
m.setObjective(gp.quicksum((cost[s, d] * x_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
m.optimize()