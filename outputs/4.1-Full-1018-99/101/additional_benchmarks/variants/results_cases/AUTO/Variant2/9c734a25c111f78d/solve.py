import gurobipy as gp
import pandas as pd
import numpy as np
supplier_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/supplier_capacity.csv'
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/customer_demand.csv'
route_variable_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_variable_costs.csv'
route_fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant2/inputs/route_fixed_costs.csv'
supplier_capacity_df = pd.read_csv(supplier_capacity_path, dtype=str, keep_default_na=False)
if 'Supplier' not in supplier_capacity_df.columns or 'SupplyCapacity' not in supplier_capacity_df.columns:
    raise KeyError("supplier_capacity.csv must contain columns 'Supplier' and 'SupplyCapacity'")
supplier_capacity_df['SupplyCapacity'] = supplier_capacity_df['SupplyCapacity'].astype(float)
suppliers = supplier_capacity_df['Supplier'].tolist()
supply_capacity = dict(zip(supplier_capacity_df['Supplier'], supplier_capacity_df['SupplyCapacity']))
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'Customer' not in customer_demand_df.columns or 'Demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'Customer' and 'Demand'")
customer_demand_df['Demand'] = customer_demand_df['Demand'].astype(float)
customers = customer_demand_df['Customer'].tolist()
demand = dict(zip(customer_demand_df['Customer'], customer_demand_df['Demand']))
route_var_costs_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
if 'Supplier' not in route_var_costs_df.columns:
    raise KeyError("route_variable_costs.csv must contain column 'Supplier'")
route_var_costs_df.set_index('Supplier', inplace=True)
route_var_costs_df = route_var_costs_df.apply(pd.to_numeric)
missing_suppliers = set(suppliers) - set(route_var_costs_df.index)
missing_customers = set(customers) - set(route_var_costs_df.columns)
if missing_suppliers:
    raise KeyError(f'route_variable_costs.csv missing suppliers: {missing_suppliers}')
if missing_customers:
    raise KeyError(f'route_variable_costs.csv missing customers: {missing_customers}')
c = {(i, j): float(route_var_costs_df.loc[i, j]) for i in suppliers for j in customers}
route_fixed_costs_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
if 'Supplier' not in route_fixed_costs_df.columns:
    raise KeyError("route_fixed_costs.csv must contain column 'Supplier'")
route_fixed_costs_df.set_index('Supplier', inplace=True)
route_fixed_costs_df = route_fixed_costs_df.apply(pd.to_numeric)
missing_suppliers_fixed = set(suppliers) - set(route_fixed_costs_df.index)
missing_customers_fixed = set(customers) - set(route_fixed_costs_df.columns)
if missing_suppliers_fixed:
    raise KeyError(f'route_fixed_costs.csv missing suppliers: {missing_suppliers_fixed}')
if missing_customers_fixed:
    raise KeyError(f'route_fixed_costs.csv missing customers: {missing_customers_fixed}')
f = {(i, j): float(route_fixed_costs_df.loc[i, j]) for i in suppliers for j in customers}
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, customers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
for i in suppliers:
    for j in customers:
        M_ij = min(supply_capacity[i], demand[j])
        m.addConstr(x_vars[i, j] <= M_ij * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()