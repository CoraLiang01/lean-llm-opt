import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant12/inputs/plant_capacity.csv'
retailer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant12/inputs/retailer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant12/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant12/inputs/route_fixed_costs.csv'
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
plant_capacity = dict(zip(plant_df['Plant'], plant_df['SupplyCapacity']))
retailer_demand = dict(zip(retailer_df['Retailer'], retailer_df['Demand']))
variable_cost = {}
fixed_cost = {}
M = {}
for i in plants:
    var_row = var_costs_df[var_costs_df['Plant'] == i]
    fixed_row = fixed_costs_df[fixed_costs_df['Plant'] == i]
    if var_row.empty or fixed_row.empty:
        raise ValueError(f'Missing cost data for plant {i}')
    for j in retailers:
        if j not in var_row.columns or j not in fixed_row.columns:
            raise ValueError(f'Missing cost column for retailer {j} in plant {i}')
        variable_cost[i, j] = float(var_row.iloc[0][j])
        fixed_cost[i, j] = float(fixed_row.iloc[0][j])
        M[i, j] = min(float(plant_capacity[i]), float(retailer_demand[j]))
m = gp.Model('FixedChargeTransportation')
x = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((variable_cost[i, j] * x[i, j] + fixed_cost[i, j] * y[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x[i, j] for i in plants)) == retailer_demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x[i, j] for j in retailers)) <= plant_capacity[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
m.optimize()