import gurobipy as gp
import pandas as pd
import numpy as np
import math
supplier_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv', dtype=str, keep_default_na=False)
if 'Supplier' not in supplier_capacity_df.columns or 'SupplyCapacity' not in supplier_capacity_df.columns:
    raise KeyError("supplier_capacity.csv must contain 'Supplier' and 'SupplyCapacity' columns.")
supplier_capacity_df['Supplier'] = supplier_capacity_df['Supplier'].str.strip()
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(float)
suppliers = supplier_capacity_df['Supplier'].tolist()
supply_capacity = supplier_capacity_df.set_index('Supplier')['SupplyCapacity'].to_dict()
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv', dtype=str, keep_default_na=False)
if 'Customer' not in customer_demand_df.columns or 'Demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain 'Customer' and 'Demand' columns.")
customer_demand_df['Customer'] = customer_demand_df['Customer'].str.strip()
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(float)
customers = customer_demand_df['Customer'].tolist()
customer_demand = customer_demand_df.set_index('Customer')['Demand'].to_dict()
route_var_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv', dtype=str, keep_default_na=False)
route_var_costs_df['Supplier'] = route_var_costs_df['Supplier'].str.strip()
for cust in customers:
    if cust not in route_var_costs_df.columns:
        raise KeyError(f"route_variable_costs.csv missing column for customer '{cust}'")
var_cost = {}
for (_, row) in route_var_costs_df.iterrows():
    i = row['Supplier']
    for j in customers:
        try:
            var_cost[i, j] = float(row[j])
        except Exception:
            raise ValueError(f"Invalid variable cost for supplier '{i}', customer '{j}' in route_variable_costs.csv")
route_fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv', dtype=str, keep_default_na=False)
route_fixed_costs_df['Supplier'] = route_fixed_costs_df['Supplier'].str.strip()
for cust in customers:
    if cust not in route_fixed_costs_df.columns:
        raise KeyError(f"route_fixed_costs.csv missing column for customer '{cust}'")
fixed_cost = {}
for (_, row) in route_fixed_costs_df.iterrows():
    i = row['Supplier']
    for j in customers:
        try:
            fixed_cost[i, j] = float(row[j])
        except Exception:
            raise ValueError(f"Invalid fixed cost for supplier '{i}', customer '{j}' in route_fixed_costs.csv")
for i in suppliers:
    for j in customers:
        if (i, j) not in var_cost:
            raise KeyError(f"Missing variable cost for supplier '{i}', customer '{j}'")
        if (i, j) not in fixed_cost:
            raise KeyError(f"Missing fixed cost for supplier '{i}', customer '{j}'")
for i in suppliers:
    if i not in supply_capacity:
        raise KeyError(f"Missing supply capacity for supplier '{i}'")
for j in customers:
    if j not in customer_demand:
        raise KeyError(f"Missing demand for customer '{j}'")
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((var_cost[i, j] * x_vars[i, j] + fixed_cost[i, j] * y_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == customer_demand[j])
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i])
for i in suppliers:
    for j in customers:
        M_ij = min(supply_capacity[i], customer_demand[j])
        m.addConstr(x_vars[i, j] <= M_ij * y_vars[i, j])
m.optimize()