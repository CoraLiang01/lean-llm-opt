import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
customers = customer_demand_df['customer'].astype(str).str.strip().tolist()
warehouses = supply_capacity_df['region'].astype(str).str.strip().tolist()
demand = {}
for (idx, row) in customer_demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand[cust] = demand_val
supply_capacity = {}
for (idx, row) in supply_capacity_df.iterrows():
    wh = str(row['region']).strip()
    try:
        cap_val = int(row['supply_capacity'])
    except Exception:
        raise ValueError(f"Invalid supply_capacity value for warehouse {wh}: {row['supply_capacity']}")
    supply_capacity[wh] = cap_val
cost = {}
cost_customer_cols = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
if set(cost_customer_cols) != set(customers):
    raise ValueError(f'Mismatch between customer columns in transportation_costs.csv and customer_demand.csv: {cost_customer_cols} vs {customers}')
for (idx, row) in transportation_costs_df.iterrows():
    wh = str(row['Unnamed: 0']).strip()
    if wh not in warehouses:
        raise ValueError(f'Warehouse {wh} in transportation_costs.csv not found in supply_capacity.csv')
    for cust in customers:
        try:
            cost_val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid cost value for warehouse {wh}, customer {cust}: {row[cust]}')
        cost[wh, cust] = cost_val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer keys between demand and customers list')
if set(supply_capacity.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse keys between supply_capacity and warehouses list')
for wh in warehouses:
    for cust in customers:
        if (wh, cust) not in cost:
            raise ValueError(f'Missing transportation cost for warehouse {wh}, customer {cust}')
m = gp.Model('GreenMart_Transportation')
x_vars = m.addVars(warehouses, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[wh, cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)), gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand[cust], name=f'demand_{cust}')
for wh in warehouses:
    m.addConstr(gp.quicksum((x_vars[wh, cust] for cust in customers)) <= supply_capacity[wh], name=f'supply_{wh}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipment Plan (warehouse -> customer: quantity) ---')
    for wh in warehouses:
        for cust in customers:
            qty = x_vars[wh, cust].X
            if qty > 1e-06:
                print(f'{wh} -> {cust}: {qty:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')