import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv', dtype=str, keep_default_na=False)
retailer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
plants = plant_capacity_df['Plant'].astype(str).str.strip().tolist()
retailers = retailer_demand_df['Retailer'].astype(str).str.strip().tolist()
plant_capacity_df['SupplyCapacity'] = plant_capacity_df['SupplyCapacity'].astype(int)
plant_capacity = dict(zip(plant_capacity_df['Plant'].astype(str).str.strip(), plant_capacity_df['SupplyCapacity']))
retailer_demand_df['Demand'] = retailer_demand_df['Demand'].astype(int)
retailer_demand = dict(zip(retailer_demand_df['Retailer'].astype(str).str.strip(), retailer_demand_df['Demand']))
route_var_costs_df['Plant'] = route_var_costs_df['Plant'].astype(str).str.strip()
c = {}
for (_, row) in route_var_costs_df.iterrows():
    i = row['Plant']
    for j in retailers:
        c[i, j] = int(row[j])
route_fixed_costs_df['Plant'] = route_fixed_costs_df['Plant'].astype(str).str.strip()
f = {}
for (_, row) in route_fixed_costs_df.iterrows():
    i = row['Plant']
    for j in retailers:
        f[i, j] = int(row[j])
M = {}
for i in plants:
    for j in retailers:
        M[i, j] = min(plant_capacity[i], retailer_demand[j])
for i in plants:
    for j in retailers:
        if (i, j) not in c or (i, j) not in f or (i, j) not in M:
            raise ValueError(f'Missing cost or M parameter for route ({i},{j})')
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