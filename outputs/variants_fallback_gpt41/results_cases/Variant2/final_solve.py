import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
supplier_capacity_path = 'benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = 'benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = 'benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = 'benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
supplier_capacity_df = pd.read_csv(supplier_capacity_path, dtype=str, keep_default_na=False)
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
route_variable_costs_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
if 'Supplier' not in supplier_capacity_df.columns:
    raise KeyError("Missing 'Supplier' column in supplier_capacity.csv")
suppliers = supplier_capacity_df['Supplier'].astype(str).tolist()
if 'Customer' not in customer_demand_df.columns:
    raise KeyError("Missing 'Customer' column in customer_demand.csv")
customers = customer_demand_df['Customer'].astype(str).tolist()
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(int)
supply_capacity = dict(zip(supplier_capacity_df['Supplier'].astype(str), supplier_capacity_df['SupplyCapacity']))
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(int)
demand = dict(zip(customer_demand_df['Customer'].astype(str), customer_demand_df['Demand']))
if 'Supplier' not in route_variable_costs_df.columns:
    raise KeyError("Missing 'Supplier' column in route_variable_costs.csv")
route_variable_costs_df = route_variable_costs_df.set_index('Supplier')
route_variable_costs_df = route_variable_costs_df.apply(pd.to_numeric, errors='raise')
c = {}
for i in suppliers:
    if i not in route_variable_costs_df.index:
        raise KeyError(f"Supplier '{i}' missing in route_variable_costs.csv")
    for j in customers:
        if j not in route_variable_costs_df.columns:
            raise KeyError(f"Customer '{j}' missing in route_variable_costs.csv columns")
        c[i, j] = int(route_variable_costs_df.loc[i, j])
if 'Supplier' not in route_fixed_costs_df.columns:
    raise KeyError("Missing 'Supplier' column in route_fixed_costs.csv")
route_fixed_costs_df = route_fixed_costs_df.set_index('Supplier')
route_fixed_costs_df = route_fixed_costs_df.apply(pd.to_numeric, errors='raise')
f = {}
for i in suppliers:
    if i not in route_fixed_costs_df.index:
        raise KeyError(f"Supplier '{i}' missing in route_fixed_costs.csv")
    for j in customers:
        if j not in route_fixed_costs_df.columns:
            raise KeyError(f"Customer '{j}' missing in route_fixed_costs.csv columns")
        f[i, j] = int(route_fixed_costs_df.loc[i, j])
M = {}
for i in suppliers:
    for j in customers:
        M[i, j] = min(supply_capacity[i], demand[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
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