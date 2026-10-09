import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
supplier_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv', dtype=str, keep_default_na=False)
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv', dtype=str, keep_default_na=False)
route_variable_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
suppliers = supplier_capacity_df['Supplier'].astype(str).str.strip().tolist()
customers = customer_demand_df['Customer'].astype(str).str.strip().tolist()
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(float)
supply_capacity = supplier_capacity_df.set_index(supplier_capacity_df['Supplier'].astype(str).str.strip())['SupplyCapacity'].to_dict()
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(float)
demand = customer_demand_df.set_index(customer_demand_df['Customer'].astype(str).str.strip())['Demand'].to_dict()
route_variable_costs_df['Supplier'] = route_variable_costs_df['Supplier'].astype(str).str.strip()
route_variable_costs_df = route_variable_costs_df.set_index('Supplier')
for cust in customers:
    if cust not in route_variable_costs_df.columns:
        raise KeyError(f"Customer '{cust}' not found in route_variable_costs.csv columns.")
for supp in suppliers:
    if supp not in route_variable_costs_df.index:
        raise KeyError(f"Supplier '{supp}' not found in route_variable_costs.csv rows.")
c = {}
for i in suppliers:
    for j in customers:
        val = route_variable_costs_df.at[i, j]
        try:
            c[i, j] = float(val)
        except Exception:
            raise ValueError(f"Invalid variable cost for route ({i},{j}): '{val}'")
route_fixed_costs_df['Supplier'] = route_fixed_costs_df['Supplier'].astype(str).str.strip()
route_fixed_costs_df = route_fixed_costs_df.set_index('Supplier')
for cust in customers:
    if cust not in route_fixed_costs_df.columns:
        raise KeyError(f"Customer '{cust}' not found in route_fixed_costs.csv columns.")
for supp in suppliers:
    if supp not in route_fixed_costs_df.index:
        raise KeyError(f"Supplier '{supp}' not found in route_fixed_costs.csv rows.")
f = {}
for i in suppliers:
    for j in customers:
        val = route_fixed_costs_df.at[i, j]
        try:
            f[i, j] = float(val)
        except Exception:
            raise ValueError(f"Invalid fixed cost for route ({i},{j}): '{val}'")
M = {}
for i in suppliers:
    for j in customers:
        M[i, j] = min(supply_capacity[i], demand[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()