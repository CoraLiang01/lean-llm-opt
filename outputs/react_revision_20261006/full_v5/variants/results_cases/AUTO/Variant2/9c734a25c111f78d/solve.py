import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
df_supcap = pd.read_csv(supplier_capacity_path, sep=',')
df_cusdemand = pd.read_csv(customer_demand_path, sep=',')
df_varcost = pd.read_csv(route_variable_costs_path, sep=',')
df_fixedcost = pd.read_csv(route_fixed_costs_path, sep=',')
df_supcap['Supplier'] = df_supcap['Supplier'].astype(str).str.strip()
df_cusdemand['Customer'] = df_cusdemand['Customer'].astype(str).str.strip()
df_varcost['Supplier'] = df_varcost['Supplier'].astype(str).str.strip()
df_fixedcost['Supplier'] = df_fixedcost['Supplier'].astype(str).str.strip()
suppliers = list(df_supcap['Supplier'].unique())
customers = list(df_cusdemand['Customer'].unique())
if not set(suppliers).issubset(set(df_varcost['Supplier'])):
    raise ValueError('Some suppliers in supplier_capacity.csv are missing from route_variable_costs.csv')
if not set(suppliers).issubset(set(df_fixedcost['Supplier'])):
    raise ValueError('Some suppliers in supplier_capacity.csv are missing from route_fixed_costs.csv')
if not set(customers).issubset(set(df_varcost.columns[1:])):
    raise ValueError('Some customers in customer_demand.csv are missing from route_variable_costs.csv columns')
if not set(customers).issubset(set(df_fixedcost.columns[1:])):
    raise ValueError('Some customers in customer_demand.csv are missing from route_fixed_costs.csv columns')
supply_capacity = df_supcap.set_index('Supplier')['SupplyCapacity'].to_dict()
demand = df_cusdemand.set_index('Customer')['Demand'].to_dict()
var_cost = {}
fixed_cost = {}
for i in suppliers:
    row_var = df_varcost[df_varcost['Supplier'] == i]
    row_fixed = df_fixedcost[df_fixedcost['Supplier'] == i]
    if row_var.empty or row_fixed.empty:
        raise ValueError(f'Supplier {i} missing in cost tables')
    for j in customers:
        if j not in row_var.columns or j not in row_fixed.columns:
            raise ValueError(f'Customer {j} missing in cost tables for supplier {i}')
        var_cost[i, j] = float(row_var.iloc[0][j])
        fixed_cost[i, j] = float(row_fixed.iloc[0][j])
routes = [(i, j) for i in suppliers for j in customers]
bigM = {(i, j): min(supply_capacity[i], demand[j]) for (i, j) in routes}
m = gp.Model('FixedChargeTransportation')
x = m.addVars(routes, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(routes, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((var_cost[i, j] * x[i, j] + fixed_cost[i, j] * y[i, j] for (i, j) in routes)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for (i, j) in routes:
    m.addConstr(x[i, j] <= bigM[i, j] * y[i, j], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for (i, j) in routes:
        print(f'{x[i, j].VarName} {x[i, j].X}')
        print(f'{y[i, j].VarName} {y[i, j].X}')
else:
    print(f'Solver status: {m.status}')