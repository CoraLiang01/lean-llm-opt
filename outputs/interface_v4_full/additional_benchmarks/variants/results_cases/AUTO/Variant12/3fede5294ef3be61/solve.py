import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv'
retailer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv'
plant_df = pd.read_csv(plant_capacity_path, sep=',')
retailer_df = pd.read_csv(retailer_demand_path, sep=',')
var_costs_df = pd.read_csv(route_variable_costs_path, sep=',')
fixed_costs_df = pd.read_csv(route_fixed_costs_path, sep=',')
plant_df['Plant'] = plant_df['Plant'].astype(str).str.strip()
retailer_df['Retailer'] = retailer_df['Retailer'].astype(str).str.strip()
var_costs_df['Plant'] = var_costs_df['Plant'].astype(str).str.strip()
fixed_costs_df['Plant'] = fixed_costs_df['Plant'].astype(str).str.strip()
plants = list(plant_df['Plant'])
retailers = list(retailer_df['Retailer'])
supply_capacity = dict(zip(plant_df['Plant'], plant_df['SupplyCapacity']))
demand = dict(zip(retailer_df['Retailer'], retailer_df['Demand']))
c = {}
for _, row in var_costs_df.iterrows():
    i = row['Plant']
    for j in retailers:
        c[i, j] = float(row[j])
f = {}
for _, row in fixed_costs_df.iterrows():
    i = row['Plant']
    for j in retailers:
        f[i, j] = float(row[j])
M = {}
for i in plants:
    for j in retailers:
        M[i, j] = min(supply_capacity[i], demand[j])
for i in plants:
    for j in retailers:
        if (i, j) not in c or (i, j) not in f or (i, j) not in M:
            raise ValueError(f'Missing cost or linking data for route ({i}, {j})')
m = gp.Model('FixedChargeTransportation')
x = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x[i, j] for j in retailers)) <= supply_capacity[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
m.optimize()