import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'.")
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customers = customer_demand_df['customer'].tolist()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(int)
customer_demand = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if 'region' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain columns 'region' and 'supply_capacity'.")
supply_capacity_df['region'] = supply_capacity_df['region'].str.strip()
warehouses = supply_capacity_df['region'].tolist()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(int)
supply_capacity = dict(zip(supply_capacity_df['region'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for warehouse IDs.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_warehouse_ids = transportation_costs_df['Unnamed: 0'].tolist()
cost_customer_ids = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
missing_warehouses = set(warehouses) - set(cost_warehouse_ids)
if missing_warehouses:
    raise ValueError(f'Warehouses {missing_warehouses} in supply_capacity.csv not found in transportation_costs.csv.')
missing_customers = set(customers) - set(cost_customer_ids)
if missing_customers:
    raise ValueError(f'Customers {missing_customers} in customer_demand.csv not found in transportation_costs.csv columns.')
cost = {}
for (idx, row) in transportation_costs_df.iterrows():
    warehouse = row['Unnamed: 0'].strip()
    for customer in customers:
        if customer not in row:
            raise KeyError(f'Customer {customer} not found in transportation_costs.csv columns.')
        try:
            cost_val = float(row[customer])
        except Exception as e:
            raise ValueError(f'Invalid cost value for warehouse {warehouse}, customer {customer}: {row[customer]}')
        cost[warehouse, customer] = cost_val
for w in warehouses:
    if w not in cost_warehouse_ids:
        raise ValueError(f'Warehouse {w} missing in transportation_costs.csv.')
for d in customers:
    if d not in cost_customer_ids:
        raise ValueError(f'Customer {d} missing in transportation_costs.csv.')
m = gp.Model('GreenMart_Transportation')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[w, d] * x_vars[w, d] for w in warehouses for d in customers)), gp.GRB.MINIMIZE)
for d in customers:
    m.addConstr(gp.quicksum((x_vars[w, d] for w in warehouses)) == customer_demand[d], name=f'demand_{d}')
for w in warehouses:
    m.addConstr(gp.quicksum((x_vars[w, d] for d in customers)) <= supply_capacity[w], name=f'supply_{w}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipping Plan ---')
    for w in warehouses:
        for d in customers:
            shipped = x_vars[w, d].X
            if shipped > 1e-06:
                print(f'Warehouse {w} -> Store {d}: {shipped:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')