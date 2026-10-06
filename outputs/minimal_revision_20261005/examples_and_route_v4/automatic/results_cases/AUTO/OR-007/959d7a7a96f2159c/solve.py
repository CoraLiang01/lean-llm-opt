import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    return str(x).strip()

def solve_greenmart_transportation():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', sep=',')
    supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', sep=',')
    cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', sep=',')
    stores = [normalize_id(c) for c in demand_df['customer']]
    warehouses = [normalize_id(r) for r in supply_df['region']]
    demand = {}
    for (idx, row) in demand_df.iterrows():
        cust = normalize_id(row['customer'])
        if cust in demand:
            raise ValueError(f'Duplicate customer in demand: {cust}')
        demand[cust] = float(row['demand'])
    supply_capacity = {}
    for (idx, row) in supply_df.iterrows():
        wh = normalize_id(row['region'])
        if wh in supply_capacity:
            raise ValueError(f'Duplicate warehouse in supply: {wh}')
        supply_capacity[wh] = float(row['supply_capacity'])
    cost_df = cost_df.rename(columns={'Unnamed: 0': 'warehouse'})
    cost_df['warehouse'] = cost_df['warehouse'].apply(normalize_id)
    cost_df.set_index('warehouse', inplace=True)
    missing_warehouses = set(warehouses) - set(cost_df.index)
    if missing_warehouses:
        raise ValueError(f'Missing warehouses in transportation_costs.csv: {missing_warehouses}')
    missing_stores = set(stores) - set(cost_df.columns)
    if missing_stores:
        raise ValueError(f'Missing stores in transportation_costs.csv: {missing_stores}')
    cost = {}
    for w in warehouses:
        for s in stores:
            try:
                c = float(cost_df.loc[w, s])
            except KeyError:
                raise ValueError(f'Missing cost entry for warehouse {w}, store {s}')
            cost[w, s] = c
    for w in warehouses:
        if w not in supply_capacity:
            raise ValueError(f'Warehouse {w} missing in supply_capacity.csv')
    for s in stores:
        if s not in demand:
            raise ValueError(f'Store {s} missing in customer_demand.csv')
    for w in warehouses:
        for s in stores:
            if (w, s) not in cost:
                raise ValueError(f'Missing cost for warehouse {w}, store {s}')
    m = gp.Model('GreenMart_Transportation')
    x_keys = [(w, s) for w in warehouses for s in stores]
    x = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[w, s] * x[w, s] for (w, s) in x_keys)), gp.GRB.MINIMIZE)
    for s in stores:
        m.addConstr(gp.quicksum((x[w, s] for w in warehouses)) == demand[s], name=f'demand_{s}')
    for w in warehouses:
        m.addConstr(gp.quicksum((x[w, s] for s in stores)) <= supply_capacity[w], name=f'supply_{w}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (w, s) in x_keys:
            print(f'{x[w, s].VarName} {x[w, s].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_greenmart_transportation()