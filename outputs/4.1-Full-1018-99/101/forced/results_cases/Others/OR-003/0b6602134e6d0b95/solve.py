import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'")
customers = customer_demand_df['customer'].astype(str).str.strip().tolist()
if 'Unnamed: 0' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'")
suppliers = supply_capacity_df['Unnamed: 0'].astype(str).str.strip().tolist()
demand = {}
for idx, row in customer_demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand[cust] = demand_val
supply_capacity = {}
for idx, row in supply_capacity_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        cap_val = int(row['supply_capacity'])
    except Exception:
        raise ValueError(f"Invalid supply_capacity value for supplier {sup}: {row['supply_capacity']}")
    supply_capacity[sup] = cap_val
cost = {}
cost_customer_cols = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
missing_customers = set(customers) - set(cost_customer_cols)
if missing_customers:
    raise KeyError(f'transportation_costs.csv missing columns for customers: {missing_customers}')
for idx, row in transportation_costs_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    if sup not in suppliers:
        raise KeyError(f'Supplier {sup} in transportation_costs.csv not found in supply_capacity.csv')
    cost[sup] = {}
    for cust in customers:
        try:
            val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        cost[sup][cust] = val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between customer_demand.csv and customers extracted')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch between supply_capacity.csv and suppliers extracted')
for sup in suppliers:
    if sup not in cost:
        raise KeyError(f'Supplier {sup} missing in transportation_costs.csv')
    for cust in customers:
        if cust not in cost[sup]:
            raise KeyError(f'Customer {cust} missing in transportation_costs.csv for supplier {sup}')
m = gp.Model('TransportationOptimization')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s][c] * x_vars[s, c] for s in suppliers for c in customers)), sense=gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_capacity_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Transportation Plan (quantities shipped) ---')
    for s in suppliers:
        for c in customers:
            val = x_vars[s, c].X
            if val > 1e-06:
                print(f'  Supplier {s} -> Customer {c}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')