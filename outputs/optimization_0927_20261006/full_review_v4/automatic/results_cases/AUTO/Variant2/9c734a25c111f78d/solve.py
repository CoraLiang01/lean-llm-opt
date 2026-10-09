import gurobipy as gp
import pandas as pd
import numpy as np
import math
supplier_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv', dtype=str, keep_default_na=False)
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(float)
suppliers = supplier_capacity_df['Supplier'].str.strip().tolist()
supply_capacity = supplier_capacity_df.set_index('Supplier')['SupplyCapacity'].to_dict()
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv', dtype=str, keep_default_na=False)
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(float)
customers = customer_demand_df['Customer'].str.strip().tolist()
customer_demand = customer_demand_df.set_index('Customer')['Demand'].to_dict()
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
cost_customer_cols = [col for col in route_var_costs_df.columns if col != 'Supplier']
for col in cost_customer_cols:
    route_var_costs_df[col] = route_var_costs_df[col].astype(float)
route_var_costs_df['Supplier'] = route_var_costs_df['Supplier'].str.strip()
c = {}
for (_, row) in route_var_costs_df.iterrows():
    i = row['Supplier']
    for j in cost_customer_cols:
        c[i, j] = float(row[j])
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
fixed_customer_cols = [col for col in route_fixed_costs_df.columns if col != 'Supplier']
for col in fixed_customer_cols:
    route_fixed_costs_df[col] = route_fixed_costs_df[col].astype(float)
route_fixed_costs_df['Supplier'] = route_fixed_costs_df['Supplier'].str.strip()
f = {}
for (_, row) in route_fixed_costs_df.iterrows():
    i = row['Supplier']
    for j in fixed_customer_cols:
        f[i, j] = float(row[j])
missing_suppliers = set(suppliers) - set(route_var_costs_df['Supplier'])
if missing_suppliers:
    raise ValueError(f'Missing suppliers in route_variable_costs.csv: {missing_suppliers}')
missing_suppliers_fixed = set(suppliers) - set(route_fixed_costs_df['Supplier'])
if missing_suppliers_fixed:
    raise ValueError(f'Missing suppliers in route_fixed_costs.csv: {missing_suppliers_fixed}')
missing_customers = set(customers) - set(cost_customer_cols)
if missing_customers:
    raise ValueError(f'Missing customers in route_variable_costs.csv: {missing_customers}')
missing_customers_fixed = set(customers) - set(fixed_customer_cols)
if missing_customers_fixed:
    raise ValueError(f'Missing customers in route_fixed_costs.csv: {missing_customers_fixed}')
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
        M_ij = min(supply_capacity[i], customer_demand[j])
        m.addConstr(x_vars[i, j] <= M_ij * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()