import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'

def solve_fixed_charge_transportation():
    df_sup = pd.read_csv(supplier_capacity_path, sep=',')
    if df_sup['Supplier'].duplicated().any():
        raise ValueError('Duplicate supplier IDs found in supplier_capacity.csv')
    suppliers = df_sup['Supplier'].astype(str).tolist()
    supply_capacity = df_sup.set_index('Supplier')['SupplyCapacity'].astype(float).to_dict()
    df_cust = pd.read_csv(customer_demand_path, sep=',')
    if df_cust['Customer'].duplicated().any():
        raise ValueError('Duplicate customer IDs found in customer_demand.csv')
    customers = df_cust['Customer'].astype(str).tolist()
    demand = df_cust.set_index('Customer')['Demand'].astype(float).to_dict()
    df_varcost = pd.read_csv(route_variable_costs_path, sep=',')
    varcost_suppliers = df_varcost['Supplier'].astype(str).tolist()
    varcost_customers = [col for col in df_varcost.columns if col != 'Supplier']
    if set(suppliers) != set(varcost_suppliers):
        raise ValueError('Mismatch in supplier IDs between supplier_capacity.csv and route_variable_costs.csv')
    if set(customers) != set(varcost_customers):
        raise ValueError('Mismatch in customer IDs between customer_demand.csv and route_variable_costs.csv')
    c = {}
    for (_, row) in df_varcost.iterrows():
        i = str(row['Supplier'])
        for j in customers:
            c[i, j] = float(row[j])
    df_fixedcost = pd.read_csv(route_fixed_costs_path, sep=',')
    fixedcost_suppliers = df_fixedcost['Supplier'].astype(str).tolist()
    fixedcost_customers = [col for col in df_fixedcost.columns if col != 'Supplier']
    if set(suppliers) != set(fixedcost_suppliers):
        raise ValueError('Mismatch in supplier IDs between supplier_capacity.csv and route_fixed_costs.csv')
    if set(customers) != set(fixedcost_customers):
        raise ValueError('Mismatch in customer IDs between customer_demand.csv and route_fixed_costs.csv')
    f = {}
    for (_, row) in df_fixedcost.iterrows():
        i = str(row['Supplier'])
        for j in customers:
            f[i, j] = float(row[j])
    pairs = [(i, j) for i in suppliers for j in customers]
    M = {(i, j): min(supply_capacity[i], demand[j]) for (i, j) in pairs}
    for (i, j) in pairs:
        if (i, j) not in c:
            raise ValueError(f'Missing variable cost for ({i},{j})')
        if (i, j) not in f:
            raise ValueError(f'Missing fixed cost for ({i},{j})')
        if i not in supply_capacity or j not in demand:
            raise ValueError(f'Missing supply or demand for ({i},{j})')
    m = gp.Model('fixed_charge_transportation')
    x = m.addVars(pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(pairs, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for (i, j) in pairs)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.addConstrs((x[i, j] <= M[i, j] * y[i, j] for (i, j) in pairs), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for (i, j) in pairs:
            print(f'x[{i},{j}] {x[i, j].VarName} {x[i, j].X}')
            print(f'y[{i},{j}] {y[i, j].VarName} {y[i, j].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_fixed_charge_transportation()