import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'")
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = customer_demand_df['customer'].tolist()
demand = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'")
supply_capacity_df['supplier'] = supply_capacity_df['Unnamed: 0'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
suppliers = supply_capacity_df['supplier'].tolist()
supply_capacity = dict(zip(supply_capacity_df['supplier'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for supplier IDs")
transportation_costs_df['supplier'] = transportation_costs_df['Unnamed: 0'].str.strip()
for c in customers:
    if c not in transportation_costs_df.columns:
        raise KeyError(f"transportation_costs.csv missing column for customer '{c}'")
cost = {}
for (idx, row) in transportation_costs_df.iterrows():
    s = row['supplier']
    for c in customers:
        try:
            cost_val = float(row[c])
        except Exception as e:
            raise ValueError(f"Invalid cost value for supplier '{s}', customer '{c}': {row[c]}")
        cost[s, c] = cost_val
for s in suppliers:
    for c in customers:
        if (s, c) not in cost:
            raise KeyError(f"Missing transportation cost for supplier '{s}', customer '{c}'")
for s in suppliers:
    if s not in supply_capacity:
        raise KeyError(f"Supplier '{s}' missing in supply_capacity")
for c in customers:
    if c not in demand:
        raise KeyError(f"Customer '{c}' missing in demand")
m = gp.Model('TransportationProblem')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
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
    print('\n--- Supplier Utilization ---')
    for s in suppliers:
        total_shipped = sum((x_vars[s, c].X for c in customers))
        print(f'  Supplier {s}: {total_shipped:.2f} / {supply_capacity[s]:.2f}')
    print('\n--- Customer Fulfillment ---')
    for c in customers:
        total_received = sum((x_vars[s, c].X for s in suppliers))
        print(f'  Customer {c}: {total_received:.2f} / {demand[c]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')