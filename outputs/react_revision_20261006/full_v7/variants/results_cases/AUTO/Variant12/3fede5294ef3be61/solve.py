import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv', dtype=str, keep_default_na=False)
retailer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv', dtype=str, keep_default_na=False)
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)

def norm_id(x):
    return x.strip()
plants = [norm_id(p) for p in plant_capacity_df['Plant']]
plant_cap_dict = {}
for (idx, row) in plant_capacity_df.iterrows():
    plant = norm_id(row['Plant'])
    try:
        cap = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f'Invalid SupplyCapacity for plant {plant}')
    plant_cap_dict[plant] = cap
retailers = [norm_id(r) for r in retailer_demand_df['Retailer']]
retailer_dem_dict = {}
for (idx, row) in retailer_demand_df.iterrows():
    retailer = norm_id(row['Retailer'])
    try:
        dem = int(row['Demand'])
    except Exception:
        raise ValueError(f'Invalid Demand for retailer {retailer}')
    retailer_dem_dict[retailer] = dem
for (df, name) in [(route_var_costs_df, 'route_variable_costs.csv'), (route_fixed_costs_df, 'route_fixed_costs.csv')]:
    file_plants = [norm_id(p) for p in df['Plant']]
    if set(file_plants) != set(plants):
        raise ValueError(f'Mismatch in plants between {name} and plant_capacity.csv')
    file_retailers = [norm_id(c) for c in df.columns if c != 'Plant']
    if set(file_retailers) != set(retailers):
        raise ValueError(f'Mismatch in retailers between {name} and retailer_demand.csv')
var_cost = {}
fixed_cost = {}
for (idx, row) in route_var_costs_df.iterrows():
    plant = norm_id(row['Plant'])
    for retailer in retailers:
        try:
            c = int(row[retailer])
        except Exception:
            raise ValueError(f'Invalid variable cost for route ({plant}, {retailer})')
        var_cost[plant, retailer] = c
for (idx, row) in route_fixed_costs_df.iterrows():
    plant = norm_id(row['Plant'])
    for retailer in retailers:
        try:
            f = int(row[retailer])
        except Exception:
            raise ValueError(f'Invalid fixed cost for route ({plant}, {retailer})')
        fixed_cost[plant, retailer] = f
M_ij = {}
for plant in plants:
    for retailer in retailers:
        M_ij[plant, retailer] = min(plant_cap_dict[plant], retailer_dem_dict[retailer])
routes = [(plant, retailer) for plant in plants for retailer in retailers]
for key in routes:
    if key not in var_cost or key not in fixed_cost or key not in M_ij:
        raise ValueError(f'Missing cost or M_ij for route {key}')

def solve_fixed_charge_transportation():
    m = gp.Model('FixedChargeTransportation')
    x_vars = m.addVars(routes, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(routes, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((var_cost[route] * x_vars[route] + fixed_cost[route] * y_vars[route] for route in routes)), gp.GRB.MINIMIZE)
    for retailer in retailers:
        m.addConstr(gp.quicksum((x_vars[plant, retailer] for plant in plants)) == retailer_dem_dict[retailer], name=f'demand_{retailer}')
    for plant in plants:
        m.addConstr(gp.quicksum((x_vars[plant, retailer] for retailer in retailers)) <= plant_cap_dict[plant], name=f'supply_{plant}')
    for route in routes:
        m.addConstr(x_vars[route] <= M_ij[route] * y_vars[route], name=f'link_{route[0]}_{route[1]}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_fixed_charge_transportation()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')