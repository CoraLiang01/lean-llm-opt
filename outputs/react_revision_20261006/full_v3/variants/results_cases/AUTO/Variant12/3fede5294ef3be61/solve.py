import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv', sep=',')
retailer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv', sep=',')
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv', sep=',')
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv', sep=',')
plants = plant_capacity_df['Plant'].astype(str).str.strip().tolist()
retailers = retailer_demand_df['Retailer'].astype(str).str.strip().tolist()
for (df, name) in [(route_var_costs_df, 'route_variable_costs.csv'), (route_fixed_costs_df, 'route_fixed_costs.csv')]:
    plant_ids = df['Plant'].astype(str).str.strip().tolist()
    if set(plant_ids) != set(plants):
        raise ValueError(f'Mismatch in plant IDs between {name} and plant_capacity.csv')
    cost_cols = [c for c in df.columns if c != 'Plant']
    if set(cost_cols) != set(retailers):
        raise ValueError(f'Mismatch in retailer IDs between {name} and retailer_demand.csv')
plant_capacity = dict(zip(plant_capacity_df['Plant'].astype(str).str.strip(), plant_capacity_df['SupplyCapacity']))
retailer_demand = dict(zip(retailer_demand_df['Retailer'].astype(str).str.strip(), retailer_demand_df['Demand']))
c = {}
f = {}
for i in plants:
    row_var = route_var_costs_df.loc[route_var_costs_df['Plant'].astype(str).str.strip() == i]
    row_fix = route_fixed_costs_df.loc[route_fixed_costs_df['Plant'].astype(str).str.strip() == i]
    if row_var.empty or row_fix.empty:
        raise ValueError(f'Missing cost data for plant {i}')
    for j in retailers:
        c[i, j] = float(row_var.iloc[0][j])
        f[i, j] = float(row_fix.iloc[0][j])
M = {}
for i in plants:
    for j in retailers:
        M[i, j] = min(plant_capacity[i], retailer_demand[j])
routes = [(i, j) for i in plants for j in retailers]

def solve_fixed_charge_transportation():
    m = gp.Model('FixedChargeTransportation')
    m.Params.MIPGap = 0.0001
    x = m.addVars(routes, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(routes, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for (i, j) in routes)), gp.GRB.MINIMIZE)
    for j in retailers:
        m.addConstr(gp.quicksum((x[i, j] for i in plants)) == retailer_demand[j], name=f'demand_{j}')
    for i in plants:
        m.addConstr(gp.quicksum((x[i, j] for j in retailers)) <= plant_capacity[i], name=f'capacity_{i}')
    for (i, j) in routes:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
    m.optimize()
    return m
m = solve_fixed_charge_transportation()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')