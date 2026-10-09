import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
df_supcap = pd.read_csv(supplier_capacity_path, sep=',')
suppliers = df_supcap['Supplier'].astype(str).tolist()
supply_capacity = dict(zip(df_supcap['Supplier'].astype(str), df_supcap['SupplyCapacity']))
df_cusdemand = pd.read_csv(customer_demand_path, sep=',')
customers = df_cusdemand['Customer'].astype(str).tolist()
customer_demand = dict(zip(df_cusdemand['Customer'].astype(str), df_cusdemand['Demand']))
df_varcost = pd.read_csv(route_variable_costs_path, sep=',')
df_varcost['Supplier'] = df_varcost['Supplier'].astype(str)
route_var_cost = {}
for (_, row) in df_varcost.iterrows():
    i = str(row['Supplier'])
    for j in customers:
        if j not in df_varcost.columns:
            raise KeyError(f'Customer {j} not found in route_variable_costs.csv columns')
        route_var_cost[i, j] = float(row[j])
df_fixedcost = pd.read_csv(route_fixed_costs_path, sep=',')
df_fixedcost['Supplier'] = df_fixedcost['Supplier'].astype(str)
route_fixed_cost = {}
for (_, row) in df_fixedcost.iterrows():
    i = str(row['Supplier'])
    for j in customers:
        if j not in df_fixedcost.columns:
            raise KeyError(f'Customer {j} not found in route_fixed_costs.csv columns')
        route_fixed_cost[i, j] = float(row[j])
if set(suppliers) != set(df_varcost['Supplier'].astype(str)):
    raise ValueError('Mismatch between suppliers in supplier_capacity.csv and route_variable_costs.csv')
if set(suppliers) != set(df_fixedcost['Supplier'].astype(str)):
    raise ValueError('Mismatch between suppliers in supplier_capacity.csv and route_fixed_costs.csv')
if set(customers) != set(df_varcost.columns[1:]):
    raise ValueError('Mismatch between customers in customer_demand.csv and route_variable_costs.csv')
if set(customers) != set(df_fixedcost.columns[1:]):
    raise ValueError('Mismatch between customers in customer_demand.csv and route_fixed_costs.csv')
M = {(i, j): min(supply_capacity[i], customer_demand[j]) for i in suppliers for j in customers}
m = gp.Model('FixedChargeTransportation')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_var_cost[i, j] * x[i, j] + route_fixed_cost[i, j] * y[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == customer_demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Ship {x[i, j].X:.2f} units from {i} to {j} (Route used: {int(y[i, j].X)})')
    print('\n--- Route Activation (y[i,j]) ---')
    for i in suppliers:
        for j in customers:
            if y[i, j].X > 0.5:
                print(f'  Route {i} -> {j}: ACTIVATED')
else:
    print(f'No optimal solution found. Status: {m.status}')