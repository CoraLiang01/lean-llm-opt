import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
supplier_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv', dtype=str, keep_default_na=False)
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(float)
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv', dtype=str, keep_default_na=False)
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(float)
route_variable_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
for col in route_variable_costs_df.columns:
    if col != 'Supplier':
        route_variable_costs_df[col] = route_variable_costs_df[col].astype(float)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
for col in route_fixed_costs_df.columns:
    if col != 'Supplier':
        route_fixed_costs_df[col] = route_fixed_costs_df[col].astype(float)
suppliers = supplier_capacity_df['Supplier'].tolist()
customers = customer_demand_df['Customer'].tolist()
route_var_suppliers = route_variable_costs_df['Supplier'].tolist()
route_fix_suppliers = route_fixed_costs_df['Supplier'].tolist()
route_var_customers = [col for col in route_variable_costs_df.columns if col != 'Supplier']
route_fix_customers = [col for col in route_fixed_costs_df.columns if col != 'Supplier']
if set(suppliers) != set(route_var_suppliers) or set(suppliers) != set(route_fix_suppliers):
    raise ValueError('Mismatch in supplier identifiers between capacity and cost tables.')
if set(customers) != set(route_var_customers) or set(customers) != set(route_fix_customers):
    raise ValueError('Mismatch in customer identifiers between demand and cost tables.')
supply_capacity = {row['Supplier']: row['SupplyCapacity'] for (_, row) in supplier_capacity_df.iterrows()}
demand = {row['Customer']: row['Demand'] for (_, row) in customer_demand_df.iterrows()}
variable_cost = {}
for (_, row) in route_variable_costs_df.iterrows():
    i = row['Supplier']
    for j in customers:
        variable_cost[i, j] = float(row[j])
fixed_cost = {}
for (_, row) in route_fixed_costs_df.iterrows():
    i = row['Supplier']
    for j in customers:
        fixed_cost[i, j] = float(row[j])
M = {}
for i in suppliers:
    for j in customers:
        M[i, j] = min(supply_capacity[i], demand[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((variable_cost[i, j] * x_vars[i, j] + fixed_cost[i, j] * y_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()