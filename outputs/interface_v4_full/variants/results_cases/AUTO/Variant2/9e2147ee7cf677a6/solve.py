import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant2/inputs/route_fixed_costs.csv'

def solve_fixed_charge_transportation():
    df_supcap = pd.read_csv(supplier_capacity_path, sep=',')
    suppliers = df_supcap['Supplier'].astype(str).tolist()
    supply_capacity = dict(zip(df_supcap['Supplier'].astype(str), df_supcap['SupplyCapacity']))
    df_custdem = pd.read_csv(customer_demand_path, sep=',')
    customers = df_custdem['Customer'].astype(str).tolist()
    customer_demand = dict(zip(df_custdem['Customer'].astype(str), df_custdem['Demand']))
    df_varcost = pd.read_csv(route_variable_costs_path, sep=',')
    df_varcost['Supplier'] = df_varcost['Supplier'].astype(str)
    df_fixcost = pd.read_csv(route_fixed_costs_path, sep=',')
    df_fixcost['Supplier'] = df_fixcost['Supplier'].astype(str)
    varcost_suppliers = set(df_varcost['Supplier'])
    fixcost_suppliers = set(df_fixcost['Supplier'])
    if set(suppliers) != varcost_suppliers or set(suppliers) != fixcost_suppliers:
        raise ValueError('Mismatch in supplier sets between capacity and cost tables.')
    varcost_customers = set(df_varcost.columns[1:])
    fixcost_customers = set(df_fixcost.columns[1:])
    if set(customers) != varcost_customers or set(customers) != fixcost_customers:
        raise ValueError('Mismatch in customer sets between demand and cost tables.')
    c = {}
    f = {}
    for _, row in df_varcost.iterrows():
        i = str(row['Supplier'])
        for j in customers:
            c[i, j] = float(row[j])
    for _, row in df_fixcost.iterrows():
        i = str(row['Supplier'])
        for j in customers:
            f[i, j] = float(row[j])
    max_supply = max(supply_capacity.values())
    max_demand = max(customer_demand.values())
    M = max(sum(supply_capacity.values()), sum(customer_demand.values()), max_supply, max_demand)
    m = gp.Model('FixedChargeTransportation')
    x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    y = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == customer_demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.addConstrs((x[i, j] <= M * y[i, j] for i in suppliers for j in customers), name='')
    m.optimize()
    return m
m = solve_fixed_charge_transportation()