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
facilities = facility_df['Facility'].tolist()
distribution_centers = demand_df['Destination'].tolist()
facility_fixed_cost = {}
facility_capacity = {}
for (idx, row) in facility_df.iterrows():
    fac = row['Facility']
    try:
        facility_fixed_cost[fac] = int(row['FixedCost'])
        facility_capacity[fac] = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in facility_costs.csv for Facility '{fac}': {e}")
demand = {}
for (idx, row) in demand_df.iterrows():
    dc = row['Destination']
    try:
        demand[dc] = int(row['Demand'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in demand_requirements.csv for Destination '{dc}': {e}")
shipping_cost = {}
for (idx, row) in shipping_df.iterrows():
    fac = row['Origin']
    if fac not in facilities:
        continue
    for dc in distribution_centers:
        if dc not in row:
            raise KeyError(f"Shipping cost column for destination '{dc}' not found in shipping_costs.csv")
        try:
            shipping_cost[fac, dc] = int(row[dc])
        except Exception as e:
            raise ValueError(f"Invalid numeric value in shipping_costs.csv for Origin '{fac}', Destination '{dc}': {e}")
if set(facilities) != set(facility_fixed_cost.keys()) or set(facilities) != set(facility_capacity.keys()):
    raise ValueError('Mismatch in facilities between facility_costs.csv and extracted keys.')
if set(distribution_centers) != set(demand.keys()):
    raise ValueError('Mismatch in distribution centers between demand_requirements.csv and extracted keys.')
for fac in facilities:
    for dc in distribution_centers:
        if (fac, dc) not in shipping_cost:
            raise ValueError(f"Missing shipping cost for facility '{fac}' to distribution center '{dc}'.")
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(facilities, distribution_centers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((facility_fixed_cost[fac] * y_vars[fac] for fac in facilities)) + gp.quicksum((shipping_cost[fac, dc] * x_vars[fac, dc] for fac in facilities for dc in distribution_centers)), gp.GRB.MINIMIZE)
for dc in distribution_centers:
    m.addConstr(gp.quicksum((x_vars[fac, dc] for fac in facilities)) == demand[dc], name=f'demand_{dc}')
for fac in facilities:
    m.addConstr(gp.quicksum((x_vars[fac, dc] for dc in distribution_centers)) <= facility_capacity[fac] * y_vars[fac], name=f'capacity_{fac}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Facility Construction Decisions ---')
    for fac in facilities:
        if y_vars[fac].X > 0.5:
            print(f'  Facility {fac}: Constructed (Fixed Cost: {facility_fixed_cost[fac]}, Capacity: {facility_capacity[fac]})')
        else:
            print(f'  Facility {fac}: Not Constructed')
    print('\n--- Shipment Plan ---')
    for fac in facilities:
        for dc in distribution_centers:
            shipped = x_vars[fac, dc].X
            if shipped > 1e-06:
                print(f'  Ship {shipped:.2f} units from {fac} to {dc} (Shipping Cost/unit: {shipping_cost[fac, dc]})')
else:
    print(f'No optimal solution found. Status: {m.status}')