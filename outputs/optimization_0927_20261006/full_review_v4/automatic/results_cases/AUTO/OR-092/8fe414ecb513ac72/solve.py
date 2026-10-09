import gurobipy as gp
import pandas as pd
import numpy as np
cost_matrix_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv'
sources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv'
destinations_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv'
cost_df = pd.read_csv(cost_matrix_path, dtype=str, keep_default_na=False)
sources_df = pd.read_csv(sources_path, dtype=str, keep_default_na=False)
dest_df = pd.read_csv(destinations_path, dtype=str, keep_default_na=False)
sources_df['source_id'] = sources_df['source_id'].str.strip()
sources = list(sources_df['source_id'])
if len(sources) != 10:
    raise ValueError(f'Expected 10 sources, got {len(sources)}')
dest_df['destination_id'] = dest_df['destination_id'].str.strip()
destinations = list(dest_df['destination_id'])
if len(destinations) != 20:
    raise ValueError(f'Expected 20 destinations, got {len(destinations)}')
sources_df['supply_units'] = sources_df['supply_units'].astype(int)
supply = dict(zip(sources_df['source_id'], sources_df['supply_units']))
dest_df['demand_units'] = dest_df['demand_units'].astype(int)
demand = dict(zip(dest_df['destination_id'], dest_df['demand_units']))
cost_df['source_id'] = cost_df['source_id'].str.strip()
cost_df = cost_df.set_index('source_id')
cost_dict = {}
for s in sources:
    if s not in cost_df.index:
        raise KeyError(f'Source {s} missing from cost matrix')
    for d in destinations:
        if d not in cost_df.columns:
            raise KeyError(f'Destination {d} missing from cost matrix columns')
        try:
            cost_dict[s, d] = float(cost_df.loc[s, d])
        except Exception as e:
            raise ValueError(f'Invalid cost for ({s},{d}): {cost_df.loc[s, d]}') from e
TRUCK_CAPACITY = 10
m = gp.Model('IntegerTruckTransportation')
t_vars = m.addVars(sources, destinations, vtype=gp.GRB.INTEGER, lb=0, name='')
f_vars = m.addVars(sources, destinations, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((cost_dict[s, d] * f_vars[s, d] for s in sources for d in destinations)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((f_vars[s, d] for d in destinations)) <= supply[s], name=f'supply_{s}')
for d in destinations:
    m.addConstr(gp.quicksum((f_vars[s, d] for s in sources)) == demand[d], name=f'demand_{d}')
for s in sources:
    for d in destinations:
        m.addConstr(f_vars[s, d] <= TRUCK_CAPACITY * t_vars[s, d], name=f'truckcap_{s}_{d}')
m.optimize()