import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
cost_df = pd.read_csv(cost_matrix_path, sep=',', dtype=str, keep_default_na=False)
dest_df = pd.read_csv(destinations_path, sep=',', dtype=str, keep_default_na=False)
src_df = pd.read_csv(sources_path, sep=',', dtype=str, keep_default_na=False)
sources = [f'S{i}' for i in range(1, 11)]
destinations = [f'D{i}' for i in range(1, 21)]
cost_df['source_id'] = cost_df['source_id'].str.strip()
cost_df.set_index('source_id', inplace=True)
cost_columns = [col for col in cost_df.columns if col.strip() in destinations]
missing_sources = set(sources) - set(cost_df.index)
missing_destinations = set(destinations) - set(cost_columns)
if missing_sources:
    raise ValueError(f'Missing sources in cost matrix: {missing_sources}')
if missing_destinations:
    raise ValueError(f'Missing destinations in cost matrix: {missing_destinations}')
cost = {}
for s in sources:
    cost[s] = {}
    for d in destinations:
        val = cost_df.at[s, d]
        try:
            cost[s][d] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for source {s}, destination {d}: {val}')
src_df['source_id'] = src_df['source_id'].str.strip()
src_df = src_df.set_index('source_id')
missing_supply_sources = set(sources) - set(src_df.index)
if missing_supply_sources:
    raise ValueError(f'Missing sources in supply table: {missing_supply_sources}')
supply = {}
for s in sources:
    val = src_df.at[s, 'supply_units']
    try:
        supply[s] = int(val)
    except Exception:
        raise ValueError(f'Invalid supply_units for source {s}: {val}')
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
dest_df = dest_df.set_index('destination_id')
missing_demand_destinations = set(destinations) - set(dest_df.index)
if missing_demand_destinations:
    raise ValueError(f'Missing destinations in demand table: {missing_demand_destinations}')
demand = {}
for d in destinations:
    val = dest_df.at[d, 'demand_units']
    try:
        demand[d] = int(val)
    except Exception:
        raise ValueError(f'Invalid demand_units for destination {d}: {val}')
m = gp.Model('Truck_Transportation')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
x_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s][d] * x_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for d in destinations:
    m.addConstr(gp.quicksum((x_vars[s, d] for s in sources)) == demand[d], name=f'demand_{d}')
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, d] for d in destinations)) <= supply[s], name=f'supply_{s}')
for s in sources:
    for d in destinations:
        m.addConstr(x_vars[s, d] <= 10 * t_vars[s, d], name=f'truck_load_{s}_{d}')
m.optimize()