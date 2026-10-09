import gurobipy as gp
import pandas as pd
import numpy as np

def solve_fixed_charge_transportation():
    supplier_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv', sep=',')
    suppliers = supplier_df['Supplier'].astype(str).tolist()
    supply_capacity = dict(zip(supplier_df['Supplier'].astype(str), supplier_df['SupplyCapacity']))
    customer_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv', sep=',')
    customers = customer_df['Customer'].astype(str).tolist()
    customer_demand = dict(zip(customer_df['Customer'].astype(str), customer_df['Demand']))
    var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv', sep=',')
    var_costs_df['Supplier'] = var_costs_df['Supplier'].astype(str)
    var_costs_df = var_costs_df.set_index('Supplier')
    if not set(suppliers).issubset(var_costs_df.index):
        missing = set(suppliers) - set(var_costs_df.index)
        raise ValueError(f'Missing suppliers in route_variable_costs.csv: {missing}')
    if not set(customers).issubset(var_costs_df.columns):
        missing = set(customers) - set(var_costs_df.columns)
        raise ValueError(f'Missing customers in route_variable_costs.csv: {missing}')
    fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv', sep=',')
    fixed_costs_df['Supplier'] = fixed_costs_df['Supplier'].astype(str)
    fixed_costs_df = fixed_costs_df.set_index('Supplier')
    if not set(suppliers).issubset(fixed_costs_df.index):
        missing = set(suppliers) - set(fixed_costs_df.index)
        raise ValueError(f'Missing suppliers in route_fixed_costs.csv: {missing}')
    if not set(customers).issubset(fixed_costs_df.columns):
        missing = set(customers) - set(fixed_costs_df.columns)
        raise ValueError(f'Missing customers in route_fixed_costs.csv: {missing}')
    route_keys = [(i, j) for i in suppliers for j in customers]
    c = {(i, j): float(var_costs_df.loc[i, j]) for (i, j) in route_keys}
    f = {(i, j): float(fixed_costs_df.loc[i, j]) for (i, j) in route_keys}
    M = {(i, j): min(supply_capacity[i], customer_demand[j]) for (i, j) in route_keys}
    m = gp.Model('FixedChargeTransportation')
    x = m.addVars(route_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(route_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for (i, j) in route_keys)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == customer_demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.addConstrs((x[i, j] <= M[i, j] * y[i, j] for (i, j) in route_keys), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for (i, j) in route_keys:
            print(f'{x[i, j].VarName} {x[i, j].X}')
            print(f'{y[i, j].VarName} {y[i, j].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_fixed_charge_transportation()