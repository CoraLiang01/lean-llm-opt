import gurobipy as gp
import pandas as pd
import numpy as np
import math
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
supplier_capacity_df = pd.read_csv(supplier_capacity_path, dtype=str, keep_default_na=False)
if 'Supplier' not in supplier_capacity_df.columns or 'SupplyCapacity' not in supplier_capacity_df.columns:
    raise KeyError("supplier_capacity.csv must contain 'Supplier' and 'SupplyCapacity' columns.")
supplier_capacity_df['Supplier'] = supplier_capacity_df['Supplier'].str.strip()
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(float)
suppliers = supplier_capacity_df['Supplier'].tolist()
supply_capacity = dict(zip(supplier_capacity_df['Supplier'], supplier_capacity_df['SupplyCapacity']))
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'Customer' not in customer_demand_df.columns or 'Demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain 'Customer' and 'Demand' columns.")
customer_demand_df['Customer'] = customer_demand_df['Customer'].str.strip()
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(float)
customers = customer_demand_df['Customer'].tolist()
customer_demand = dict(zip(customer_demand_df['Customer'], customer_demand_df['Demand']))
route_var_costs_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
if 'Supplier' not in route_var_costs_df.columns:
    raise KeyError("route_variable_costs.csv must contain 'Supplier' column.")
route_var_costs_df['Supplier'] = route_var_costs_df['Supplier'].str.strip()
for cust in customers:
    if cust not in route_var_costs_df.columns:
        raise KeyError(f"route_variable_costs.csv missing column for customer '{cust}'")
route_var_costs = {}
for _, row in route_var_costs_df.iterrows():
    i = row['Supplier']
    for j in customers:
        try:
            route_var_costs[i, j] = float(row[j])
        except Exception:
            raise ValueError(f'Invalid variable cost for route ({i},{j}) in route_variable_costs.csv')
route_fixed_costs_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
if 'Supplier' not in route_fixed_costs_df.columns:
    raise KeyError("route_fixed_costs.csv must contain 'Supplier' column.")
route_fixed_costs_df['Supplier'] = route_fixed_costs_df['Supplier'].str.strip()
for cust in customers:
    if cust not in route_fixed_costs_df.columns:
        raise KeyError(f"route_fixed_costs.csv missing column for customer '{cust}'")
route_fixed_costs = {}
for _, row in route_fixed_costs_df.iterrows():
    i = row['Supplier']
    for j in customers:
        try:
            route_fixed_costs[i, j] = float(row[j])
        except Exception:
            raise ValueError(f'Invalid fixed cost for route ({i},{j}) in route_fixed_costs.csv')
for i in suppliers:
    for j in customers:
        if (i, j) not in route_var_costs:
            raise KeyError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in route_fixed_costs:
            raise KeyError(f'Missing fixed cost for route ({i},{j})')
bigM = {(i, j): min(supply_capacity[i], customer_demand[j]) for i in suppliers for j in customers}
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_costs[i, j] * x_vars[i, j] + route_fixed_costs[i, j] * y_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == customer_demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= bigM[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()