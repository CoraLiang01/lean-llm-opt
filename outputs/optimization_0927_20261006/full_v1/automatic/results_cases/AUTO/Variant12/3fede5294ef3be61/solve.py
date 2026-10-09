import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
plant_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv', dtype=str, keep_default_na=False)
retailer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
plants = plant_capacity_df['Plant'].astype(str).str.strip().tolist()
retailers = retailer_demand_df['Retailer'].astype(str).str.strip().tolist()
plant_capacity = {}
for (_, row) in plant_capacity_df.iterrows():
    plant = str(row['Plant']).strip()
    try:
        plant_capacity[plant] = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f"Invalid SupplyCapacity for plant {plant}: {row['SupplyCapacity']}")
retailer_demand = {}
for (_, row) in retailer_demand_df.iterrows():
    retailer = str(row['Retailer']).strip()
    try:
        retailer_demand[retailer] = int(row['Demand'])
    except Exception:
        raise ValueError(f"Invalid Demand for retailer {retailer}: {row['Demand']}")
route_var_costs_df = route_var_costs_df.copy()
route_var_costs_df['Plant'] = route_var_costs_df['Plant'].astype(str).str.strip()
c = {}
for (_, row) in route_var_costs_df.iterrows():
    plant = str(row['Plant']).strip()
    for retailer in retailers:
        try:
            c[plant, retailer] = float(row[retailer])
        except Exception:
            raise ValueError(f'Missing or invalid variable cost for route ({plant}, {retailer})')
route_fixed_costs_df = route_fixed_costs_df.copy()
route_fixed_costs_df['Plant'] = route_fixed_costs_df['Plant'].astype(str).str.strip()
f = {}
for (_, row) in route_fixed_costs_df.iterrows():
    plant = str(row['Plant']).strip()
    for retailer in retailers:
        try:
            f[plant, retailer] = float(row[retailer])
        except Exception:
            raise ValueError(f'Missing or invalid fixed cost for route ({plant}, {retailer})')
M = {}
for i in plants:
    for j in retailers:
        if i not in plant_capacity:
            raise KeyError(f'Plant {i} missing from plant_capacity')
        if j not in retailer_demand:
            raise KeyError(f'Retailer {j} missing from retailer_demand')
        M[i, j] = min(plant_capacity[i], retailer_demand[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == retailer_demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in retailers)) <= plant_capacity[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()