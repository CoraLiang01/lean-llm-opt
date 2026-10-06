import gurobipy as gp
import pandas as pd
import numpy as np

def solve_fixed_charge_transportation():
    plant_cap_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv'
    retailer_dem_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv'
    route_var_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv'
    route_fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv'
    df_plant = pd.read_csv(plant_cap_path, sep=',')
    df_plant['Plant'] = df_plant['Plant'].astype(str).str.strip()
    plants = df_plant['Plant'].unique().tolist()
    supply_capacity = dict(zip(df_plant['Plant'], df_plant['SupplyCapacity']))
    df_retailer = pd.read_csv(retailer_dem_path, sep=',')
    df_retailer['Retailer'] = df_retailer['Retailer'].astype(str).str.strip()
    retailers = df_retailer['Retailer'].unique().tolist()
    demand = dict(zip(df_retailer['Retailer'], df_retailer['Demand']))
    df_var_cost = pd.read_csv(route_var_cost_path, sep=',')
    df_var_cost['Plant'] = df_var_cost['Plant'].astype(str).str.strip()
    if not set(plants).issubset(set(df_var_cost['Plant'])):
        raise ValueError('Some plants in plant_capacity.csv are missing in route_variable_costs.csv')
    if not set(retailers).issubset(set(df_var_cost.columns[1:])):
        raise ValueError('Some retailers in retailer_demand.csv are missing in route_variable_costs.csv')
    c = {}
    for (_, row) in df_var_cost.iterrows():
        i = str(row['Plant']).strip()
        for j in retailers:
            c[i, j] = float(row[j])
    df_fixed_cost = pd.read_csv(route_fixed_cost_path, sep=',')
    df_fixed_cost['Plant'] = df_fixed_cost['Plant'].astype(str).str.strip()
    if not set(plants).issubset(set(df_fixed_cost['Plant'])):
        raise ValueError('Some plants in plant_capacity.csv are missing in route_fixed_costs.csv')
    if not set(retailers).issubset(set(df_fixed_cost.columns[1:])):
        raise ValueError('Some retailers in retailer_demand.csv are missing in route_fixed_costs.csv')
    f = {}
    for (_, row) in df_fixed_cost.iterrows():
        i = str(row['Plant']).strip()
        for j in retailers:
            f[i, j] = float(row[j])
    routes = [(i, j) for i in plants for j in retailers]
    M = {}
    for (i, j) in routes:
        M[i, j] = min(supply_capacity[i], demand[j])
    for (i, j) in routes:
        if (i, j) not in c:
            raise ValueError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in f:
            raise ValueError(f'Missing fixed cost for route ({i},{j})')
        if i not in supply_capacity or j not in demand:
            raise ValueError(f'Missing supply or demand for ({i},{j})')
    m = gp.Model('FixedChargeTransportation')
    x = m.addVars(routes, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(routes, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for (i, j) in routes)), gp.GRB.MINIMIZE)
    for j in retailers:
        m.addConstr(gp.quicksum((x[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
    for i in plants:
        m.addConstr(gp.quicksum((x[i, j] for j in retailers)) <= supply_capacity[i], name=f'supply_{i}')
    for (i, j) in routes:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for (i, j) in routes:
            print(f'{x[i, j].VarName} {x[i, j].X}')
            print(f'{y[i, j].VarName} {y[i, j].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_fixed_charge_transportation()