import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
facility_df = pd.read_csv(facility_costs_path, sep=',')
shipping_df = pd.read_csv(shipping_costs_path, sep=',')
demand_df = pd.read_csv(demand_requirements_path, sep=',')
facilities = [f'A{i}' for i in range(1, 16)]
distribution_centers = [f'B{j}' for j in range(1, 9)]
facility_set = set(facility_df['Facility'].astype(str).str.strip())
if set(facilities) != facility_set:
    missing = set(facilities) - facility_set
    extra = facility_set - set(facilities)
    raise ValueError(f'Facility mismatch: missing {missing}, extra {extra}')
dc_set = set(demand_df['Destination'].astype(str).str.strip())
if set(distribution_centers) != dc_set:
    missing = set(distribution_centers) - dc_set
    extra = dc_set - set(distribution_centers)
    raise ValueError(f'Distribution center mismatch: missing {missing}, extra {extra}')
facility_df['Facility'] = facility_df['Facility'].astype(str).str.strip()
fixed_cost = facility_df.set_index('Facility')['FixedCost'].to_dict()
capacity = facility_df.set_index('Facility')['Capacity'].to_dict()
demand_df['Destination'] = demand_df['Destination'].astype(str).str.strip()
demand = demand_df.set_index('Destination')['Demand'].to_dict()
shipping_df['Origin'] = shipping_df['Origin'].astype(str).str.strip()
shipping_cost = {}
for i in facilities:
    row = shipping_df.loc[shipping_df['Origin'] == i]
    if row.empty:
        raise ValueError(f'Shipping cost row missing for facility {i}')
    for j in distribution_centers:
        if j not in row.columns:
            raise ValueError(f'Shipping cost column missing for DC {j}')
        shipping_cost[i, j] = float(row.iloc[0][j])
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x = m.addVars(facilities, distribution_centers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x[i, j] for i in facilities for j in distribution_centers)), gp.GRB.MINIMIZE)
for j in distribution_centers:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    m.addConstr(gp.quicksum((x[i, j] for j in distribution_centers)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Facility Construction Decisions ---')
    for i in facilities:
        if y[i].X > 0.5:
            print(f'  Facility {i}: BUILT (FixedCost={fixed_cost[i]}, Capacity={capacity[i]})')
        else:
            print(f'  Facility {i}: NOT BUILT')
    print('\n--- Shipment Plan (facility -> DC: amount) ---')
    for i in facilities:
        for j in distribution_centers:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')