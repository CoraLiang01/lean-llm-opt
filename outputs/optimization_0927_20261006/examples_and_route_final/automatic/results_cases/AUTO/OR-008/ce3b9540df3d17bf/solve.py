import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
customers = customer_demand_df['Customers'].astype(str).str.strip().tolist()
demand = {}
for (idx, row) in customer_demand_df.iterrows():
    cust = str(row['Customers']).strip()
    try:
        demand[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer '{cust}': {row['demand']}") from e
suppliers = supply_capacity_df['Suppliers'].astype(str).str.strip().tolist()
supply_capacity = {}
for (idx, row) in supply_capacity_df.iterrows():
    sup = str(row['Suppliers']).strip()
    try:
        supply_capacity[sup] = int(row['supply_capacity'])
    except Exception as e:
        raise ValueError(f"Invalid supply_capacity value for supplier '{sup}': {row['supply_capacity']}") from e
cost = {}
cost_supplier_names = transportation_costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
cost_customer_names = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
missing_suppliers = set(suppliers) - set(cost_supplier_names)
if missing_suppliers:
    raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
missing_customers = set(customers) - set(cost_customer_names)
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
for (i, row) in transportation_costs_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    if sup not in suppliers:
        continue
    for cust in customers:
        try:
            val = row[cust]
            cost_val = int(val)
        except Exception as e:
            raise ValueError(f"Invalid cost value for supplier '{sup}', customer '{cust}': {val}") from e
        cost[sup, cust] = cost_val
m = gp.Model('FreshMart_Transportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name='')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand[c], name='')
m.optimize()