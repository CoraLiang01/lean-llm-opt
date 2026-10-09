import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
facility_df = pd.read_csv(facility_costs_path, dtype=str, keep_default_na=False)
shipping_df = pd.read_csv(shipping_costs_path, dtype=str, keep_default_na=False)
demand_df = pd.read_csv(demand_requirements_path, dtype=str, keep_default_na=False)
facilities = [f'A{i}' for i in range(1, 16)]
distribution_centers = [f'B{j}' for j in range(1, 9)]
facility_ids_in_data = set(facility_df['Facility'].str.strip())
if set(facilities) - facility_ids_in_data:
    missing = set(facilities) - facility_ids_in_data
    raise ValueError(f'Missing facilities in facility_costs.csv: {missing}')
shipping_origins_in_data = set(shipping_df['Origin'].str.strip())
if set(facilities) - shipping_origins_in_data:
    missing = set(facilities) - shipping_origins_in_data
    raise ValueError(f'Missing facilities in shipping_costs.csv: {missing}')
shipping_dest_cols = [col for col in shipping_df.columns if col != 'Origin']
if set(distribution_centers) - set(shipping_dest_cols):
    missing = set(distribution_centers) - set(shipping_dest_cols)
    raise ValueError(f'Missing distribution centers in shipping_costs.csv: {missing}')
demand_dest_in_data = set(demand_df['Destination'].str.strip())
if set(distribution_centers) - demand_dest_in_data:
    missing = set(distribution_centers) - demand_dest_in_data
    raise ValueError(f'Missing distribution centers in demand_requirements.csv: {missing}')
facility_df = facility_df.set_index('Facility')
facility_df.index = facility_df.index.str.strip()
fixed_cost = {i: int(facility_df.loc[i, 'FixedCost']) for i in facilities}
capacity = {i: int(facility_df.loc[i, 'Capacity']) for i in facilities}
demand_df = demand_df.set_index('Destination')
demand_df.index = demand_df.index.str.strip()
demand = {j: int(demand_df.loc[j, 'Demand']) for j in distribution_centers}
shipping_df = shipping_df.set_index('Origin')
shipping_df.index = shipping_df.index.str.strip()
shipping_cost = {}
for i in facilities:
    for j in distribution_centers:
        val = shipping_df.loc[i, j]
        try:
            shipping_cost[i, j] = float(val)
        except Exception:
            raise ValueError(f'Invalid shipping cost for ({i},{j}): {val}')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(facilities, distribution_centers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x_vars[i, j] for i in facilities for j in distribution_centers)), gp.GRB.MINIMIZE)
for j in distribution_centers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in distribution_centers)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()