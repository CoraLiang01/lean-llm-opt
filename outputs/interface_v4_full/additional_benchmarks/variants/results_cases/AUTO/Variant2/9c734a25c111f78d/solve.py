import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
df_supply = pd.read_csv(supplier_capacity_path, sep=',')
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_varcost = pd.read_csv(route_variable_costs_path, sep=',')
df_fixedcost = pd.read_csv(route_fixed_costs_path, sep=',')
df_supply['Supplier'] = df_supply['Supplier'].astype(str).str.strip()
df_demand['Customer'] = df_demand['Customer'].astype(str).str.strip()
df_varcost['Supplier'] = df_varcost['Supplier'].astype(str).str.strip()
df_fixedcost['Supplier'] = df_fixedcost['Supplier'].astype(str).str.strip()
suppliers = list(df_supply['Supplier'])
customers = list(df_demand['Customer'])
if not set(suppliers) <= set(df_varcost['Supplier']):
    raise ValueError('Some suppliers in supplier_capacity.csv are missing from route_variable_costs.csv')
if not set(suppliers) <= set(df_fixedcost['Supplier']):
    raise ValueError('Some suppliers in supplier_capacity.csv are missing from route_fixed_costs.csv')
if not set(customers) <= set(df_varcost.columns[1:]):
    raise ValueError('Some customers in customer_demand.csv are missing from route_variable_costs.csv columns')
if not set(customers) <= set(df_fixedcost.columns[1:]):
    raise ValueError('Some customers in customer_demand.csv are missing from route_fixed_costs.csv columns')
supply_capacity = df_supply.set_index('Supplier')['SupplyCapacity'].to_dict()
demand = df_demand.set_index('Customer')['Demand'].to_dict()
c = {}
f = {}
for i in suppliers:
    row_var = df_varcost[df_varcost['Supplier'] == i]
    row_fix = df_fixedcost[df_fixedcost['Supplier'] == i]
    if row_var.empty or row_fix.empty:
        raise ValueError(f'Supplier {i} missing in cost tables')
    for j in customers:
        c[i, j] = float(row_var.iloc[0][j])
        f[i, j] = float(row_fix.iloc[0][j])
M = {(i, j): min(supply_capacity[i], demand[j]) for i in suppliers for j in customers}
m = gp.Model('FixedChargeTransportation')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
m.optimize()