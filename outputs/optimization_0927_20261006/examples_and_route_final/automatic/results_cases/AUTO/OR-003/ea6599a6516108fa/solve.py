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
customers = customer_demand_df['customer'].astype(str).str.strip().tolist()
suppliers = supply_capacity_df['Unnamed: 0'].astype(str).str.strip().tolist()
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
    sup = str(row['Unnamed: 0']).strip()
    try:
        cap_val = int(row['supply_capacity'])
    except Exception:
        raise ValueError(f"Invalid supply_capacity value for supplier {sup}: {row['supply_capacity']}")
    supply_capacity[sup] = cap_val
cost = {}
missing_customers = [c for c in customers if c not in transportation_costs_df.columns]
if missing_customers:
    raise KeyError(f'Missing customer columns in transportation_costs.csv: {missing_customers}')
for (idx, row) in transportation_costs_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    if sup not in suppliers:
        continue
    for cust in customers:
        try:
            cost_val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid cost value for supplier {sup}, customer {cust}: {row[cust]}')
        cost[sup, cust] = cost_val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between customer_demand.csv and customers list.')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch between supply_capacity.csv and suppliers list.')
for s in suppliers:
    for c in customers:
        if (s, c) not in cost:
            raise KeyError(f'Missing transportation cost for supplier {s}, customer {c}.')
m = gp.Model('TransportationProblem')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Transportation Plan (units shipped) ---')
    for s in suppliers:
        for c in customers:
            shipped = x_vars[s, c].X
            if shipped > 1e-06:
                print(f'  Supplier {s} -> Customer {c}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')