import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv', dtype=str, keep_default_na=False)
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv', dtype=str, keep_default_na=False)
route_variable_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
suppliers = supplier_capacity_df['Supplier'].astype(str).str.strip().tolist()
customers = customer_demand_df['Customer'].astype(str).str.strip().tolist()
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(float)
supply_capacity = dict(zip(supplier_capacity_df['Supplier'].astype(str).str.strip(), supplier_capacity_df['SupplyCapacity']))
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(float)
customer_demand = dict(zip(customer_demand_df['Customer'].astype(str).str.strip(), customer_demand_df['Demand']))
route_variable_costs_df['Supplier'] = route_variable_costs_df['Supplier'].astype(str).str.strip()
route_variable_costs_df = route_variable_costs_df.set_index('Supplier')
for cust in customers:
    if cust not in route_variable_costs_df.columns:
        raise KeyError(f"Customer '{cust}' not found in route_variable_costs.csv columns.")
route_variable_costs_df = route_variable_costs_df[customers].apply(pd.to_numeric)
c = {(i, j): float(route_variable_costs_df.loc[i, j]) for i in suppliers for j in customers}
route_fixed_costs_df['Supplier'] = route_fixed_costs_df['Supplier'].astype(str).str.strip()
route_fixed_costs_df = route_fixed_costs_df.set_index('Supplier')
for cust in customers:
    if cust not in route_fixed_costs_df.columns:
        raise KeyError(f"Customer '{cust}' not found in route_fixed_costs.csv columns.")
route_fixed_costs_df = route_fixed_costs_df[customers].apply(pd.to_numeric)
f = {(i, j): float(route_fixed_costs_df.loc[i, j]) for i in suppliers for j in customers}
M = {(i, j): min(supply_capacity[i], customer_demand[j]) for i in suppliers for j in customers}
for i in suppliers:
    if i not in supply_capacity:
        raise KeyError(f"Supplier '{i}' missing in supply_capacity.")
for j in customers:
    if j not in customer_demand:
        raise KeyError(f"Customer '{j}' missing in customer_demand.")
for i in suppliers:
    for j in customers:
        if (i, j) not in c:
            raise KeyError(f'Missing variable cost for ({i},{j})')
        if (i, j) not in f:
            raise KeyError(f'Missing fixed cost for ({i},{j})')
        if (i, j) not in M:
            raise KeyError(f'Missing big-M for ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == customer_demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()