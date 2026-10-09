import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
supplier_capacity_df = pd.read_csv(supplier_capacity_path, dtype=str, keep_default_na=False)
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
route_variable_costs_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
route_fixed_costs_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
suppliers = supplier_capacity_df['Supplier'].str.strip().tolist()
customers = customer_demand_df['Customer'].str.strip().tolist()
supply_capacity = {}
for (idx, row) in supplier_capacity_df.iterrows():
    supplier = str(row['Supplier']).strip()
    try:
        supply_capacity[supplier] = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f"Invalid SupplyCapacity for supplier {supplier}: {row['SupplyCapacity']}")
demand = {}
for (idx, row) in customer_demand_df.iterrows():
    customer = str(row['Customer']).strip()
    try:
        demand[customer] = int(row['Demand'])
    except Exception:
        raise ValueError(f"Invalid Demand for customer {customer}: {row['Demand']}")
route_variable_costs = {}
for (idx, row) in route_variable_costs_df.iterrows():
    supplier = str(row['Supplier']).strip()
    for customer in customers:
        if customer not in row:
            raise KeyError(f'Customer {customer} not found in route_variable_costs.csv columns')
        try:
            route_variable_costs[supplier, customer] = float(row[customer])
        except Exception:
            raise ValueError(f'Invalid variable cost for route ({supplier},{customer}): {row[customer]}')
route_fixed_costs = {}
for (idx, row) in route_fixed_costs_df.iterrows():
    supplier = str(row['Supplier']).strip()
    for customer in customers:
        if customer not in row:
            raise KeyError(f'Customer {customer} not found in route_fixed_costs.csv columns')
        try:
            route_fixed_costs[supplier, customer] = float(row[customer])
        except Exception:
            raise ValueError(f'Invalid fixed cost for route ({supplier},{customer}): {row[customer]}')
bigM = {}
for i in suppliers:
    for j in customers:
        if i not in supply_capacity:
            raise KeyError(f'Supplier {i} missing from supply_capacity')
        if j not in demand:
            raise KeyError(f'Customer {j} missing from demand')
        bigM[i, j] = min(supply_capacity[i], demand[j])
for i in suppliers:
    for j in customers:
        if (i, j) not in route_variable_costs:
            raise KeyError(f'Missing variable cost for ({i},{j})')
        if (i, j) not in route_fixed_costs:
            raise KeyError(f'Missing fixed cost for ({i},{j})')
        if (i, j) not in bigM:
            raise KeyError(f'Missing big-M for ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_variable_costs[i, j] * x_vars[i, j] + route_fixed_costs[i, j] * y_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= bigM[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()