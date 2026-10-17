import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv', dtype=str, keep_default_na=False)
if 'Plant' not in plant_capacity_df.columns or 'SupplyCapacity' not in plant_capacity_df.columns:
    raise KeyError("plant_capacity.csv must contain columns 'Plant' and 'SupplyCapacity'")
plant_capacity_df['SupplyCapacity'] = plant_capacity_df['SupplyCapacity'].astype(int)
plants = plant_capacity_df['Plant'].tolist()
plant_cap = dict(zip(plant_capacity_df['Plant'], plant_capacity_df['SupplyCapacity']))
retailer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv', dtype=str, keep_default_na=False)
if 'Retailer' not in retailer_demand_df.columns or 'Demand' not in retailer_demand_df.columns:
    raise KeyError("retailer_demand.csv must contain columns 'Retailer' and 'Demand'")
retailer_demand_df['Demand'] = retailer_demand_df['Demand'].astype(int)
retailers = retailer_demand_df['Retailer'].tolist()
retailer_dem = dict(zip(retailer_demand_df['Retailer'], retailer_demand_df['Demand']))
route_varcosts_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
if 'Plant' not in route_varcosts_df.columns:
    raise KeyError("route_variable_costs.csv must contain column 'Plant'")
for r in retailers:
    if r not in route_varcosts_df.columns:
        raise KeyError(f"route_variable_costs.csv missing retailer column '{r}'")
route_varcosts_df.set_index('Plant', inplace=True)
route_varcosts_df = route_varcosts_df.apply(pd.to_numeric)
var_cost = {(i, j): route_varcosts_df.at[i, j] for i in plants for j in retailers}
route_fixedcosts_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
if 'Plant' not in route_fixedcosts_df.columns:
    raise KeyError("route_fixed_costs.csv must contain column 'Plant'")
for r in retailers:
    if r not in route_fixedcosts_df.columns:
        raise KeyError(f"route_fixed_costs.csv missing retailer column '{r}'")
route_fixedcosts_df.set_index('Plant', inplace=True)
route_fixedcosts_df = route_fixedcosts_df.apply(pd.to_numeric)
fixed_cost = {(i, j): route_fixedcosts_df.at[i, j] for i in plants for j in retailers}
M = {(i, j): min(plant_cap[i], retailer_dem[j]) for i in plants for j in retailers}
for i in plants:
    for j in retailers:
        if (i, j) not in var_cost or (i, j) not in fixed_cost or (i, j) not in M:
            raise KeyError(f'Missing cost or M parameter for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((var_cost[i, j] * x_vars[i, j] + fixed_cost[i, j] * y_vars[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == retailer_dem[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in retailers)) <= plant_cap[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()