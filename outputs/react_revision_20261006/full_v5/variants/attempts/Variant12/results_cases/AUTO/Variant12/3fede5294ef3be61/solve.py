import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv', sep=',')
retailer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv', sep=',')
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv', sep=',')
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv', sep=',')
plant_capacity_df['Plant'] = plant_capacity_df['Plant'].astype(str).str.strip()
retailer_demand_df['Retailer'] = retailer_demand_df['Retailer'].astype(str).str.strip()
route_var_costs_df['Plant'] = route_var_costs_df['Plant'].astype(str).str.strip()
route_fixed_costs_df['Plant'] = route_fixed_costs_df['Plant'].astype(str).str.strip()
plants = list(plant_capacity_df['Plant'])
retailers = list(retailer_demand_df['Retailer'])
plant_capacity = dict(zip(plant_capacity_df['Plant'], plant_capacity_df['SupplyCapacity']))
retailer_demand = dict(zip(retailer_demand_df['Retailer'], retailer_demand_df['Demand']))
variable_cost = {}
fixed_cost = {}
M = {}
for i in plants:
    if i not in route_var_costs_df['Plant'].values:
        raise ValueError(f'Plant {i} missing in route_variable_costs.csv')
    if i not in route_fixed_costs_df['Plant'].values:
        raise ValueError(f'Plant {i} missing in route_fixed_costs.csv')
    var_row = route_var_costs_df[route_var_costs_df['Plant'] == i].iloc[0]
    fix_row = route_fixed_costs_df[route_fixed_costs_df['Plant'] == i].iloc[0]
    for j in retailers:
        if j not in route_var_costs_df.columns or j not in route_fixed_costs_df.columns:
            raise ValueError(f'Retailer {j} missing in cost tables')
        variable_cost[i, j] = float(var_row[j])
        fixed_cost[i, j] = float(fix_row[j])
        M[i, j] = min(float(plant_capacity[i]), float(retailer_demand[j]))
routes = [(i, j) for i in plants for j in retailers]

def solve_fixed_charge_transportation():
    m = gp.Model('FixedChargeTransportation')
    x = m.addVars(routes, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(routes, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((variable_cost[i, j] * x[i, j] + fixed_cost[i, j] * y[i, j] for (i, j) in routes)), gp.GRB.MINIMIZE)
    for j in retailers:
        m.addConstr(gp.quicksum((x[i, j] for i in plants)) == retailer_demand[j], name=f'demand_{j}')
    for i in plants:
        m.addConstr(gp.quicksum((x[i, j] for j in retailers)) <= plant_capacity[i], name=f'capacity_{i}')
    for (i, j) in routes:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
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