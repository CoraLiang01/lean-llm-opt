import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
supplier_df = pd.read_csv(supplier_capacity_path, dtype=str, keep_default_na=False)
supplier_df['SupplyCapacity'] = supplier_df['SupplyCapacity'].astype(float)
suppliers = supplier_df['Supplier'].tolist()
supply_capacity = dict(zip(supplier_df['Supplier'], supplier_df['SupplyCapacity']))
customer_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
customer_df['Demand'] = customer_df['Demand'].astype(float)
customers = customer_df['Customer'].tolist()
customer_demand = dict(zip(customer_df['Customer'], customer_df['Demand']))
route_varcosts_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
cost_customer_cols = [col for col in route_varcosts_df.columns if col != 'Supplier']
if set(cost_customer_cols) != set(customers):
    raise ValueError(f'Mismatch between customers in route_variable_costs.csv and customer_demand.csv: {cost_customer_cols} vs {customers}')
route_varcosts_df.set_index('Supplier', inplace=True)
route_varcosts_df = route_varcosts_df.apply(pd.to_numeric)
route_varcosts = {}
for i in suppliers:
    for j in customers:
        route_varcosts[i, j] = float(route_varcosts_df.loc[i, j])
route_fixedcosts_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
fixed_customer_cols = [col for col in route_fixedcosts_df.columns if col != 'Supplier']
if set(fixed_customer_cols) != set(customers):
    raise ValueError(f'Mismatch between customers in route_fixed_costs.csv and customer_demand.csv: {fixed_customer_cols} vs {customers}')
route_fixedcosts_df.set_index('Supplier', inplace=True)
route_fixedcosts_df = route_fixedcosts_df.apply(pd.to_numeric)
route_fixedcosts = {}
for i in suppliers:
    for j in customers:
        route_fixedcosts[i, j] = float(route_fixedcosts_df.loc[i, j])
route_keys = [(i, j) for i in suppliers for j in customers]
M = {}
for i in suppliers:
    for j in customers:
        M[i, j] = min(supply_capacity[i], customer_demand[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(route_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(route_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((route_varcosts[i, j] * x_vars[i, j] + route_fixedcosts[i, j] * y_vars[i, j] for (i, j) in route_keys)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == customer_demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for (i, j) in route_keys:
    m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for (i, j) in route_keys:
        print(f'{x_vars[i, j].VarName} {x_vars[i, j].X}')
        print(f'{y_vars[i, j].VarName} {y_vars[i, j].X}')
else:
    print(f'Solver status: {m.status}')