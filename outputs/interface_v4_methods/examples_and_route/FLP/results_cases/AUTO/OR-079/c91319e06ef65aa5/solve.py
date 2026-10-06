import gurobipy as gp
import pandas as pd
import numpy as np
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
facility_df = pd.read_csv(facility_costs_path, sep=',')
facility_df['Facility'] = facility_df['Facility'].astype(str).str.strip()
facilities = facility_df['Facility'].tolist()
fixed_cost = facility_df.set_index('Facility')['FixedCost'].to_dict()
capacity = facility_df.set_index('Facility')['Capacity'].to_dict()
demand_df = pd.read_csv(demand_requirements_path, sep=',')
demand_df['Destination'] = demand_df['Destination'].astype(str).str.strip()
distribution_centers = demand_df['Destination'].tolist()
demand = demand_df.set_index('Destination')['Demand'].to_dict()
shipping_df = pd.read_csv(shipping_costs_path, sep=',')
shipping_df['Origin'] = shipping_df['Origin'].astype(str).str.strip()
shipping_cost = {}
for i in facilities:
    row = shipping_df.loc[shipping_df['Origin'] == i]
    if row.empty:
        raise ValueError(f"Shipping cost row missing for facility '{i}'")
    for j in distribution_centers:
        if j not in row.columns:
            raise ValueError(f"Shipping cost column missing for destination '{j}'")
        shipping_cost[i, j] = float(row.iloc[0][j])
if set(facilities) != set(fixed_cost.keys()) or set(facilities) != set(capacity.keys()):
    raise ValueError('Mismatch in facilities between cost/capacity tables and facility list.')
if set(distribution_centers) != set(demand.keys()):
    raise ValueError('Mismatch in distribution centers between demand table and DC list.')
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x = m.addVars(facilities, distribution_centers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x[i, j] for i in facilities for j in distribution_centers)), gp.GRB.MINIMIZE)
for j in distribution_centers:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    m.addConstr(gp.quicksum((x[i, j] for j in distribution_centers)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()