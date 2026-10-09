import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'.")
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customers = customer_demand_df['customer'].tolist()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(int)
demand_dict = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'.")
supply_capacity_df['Unnamed: 0'] = supply_capacity_df['Unnamed: 0'].str.strip()
supplies = supply_capacity_df['Unnamed: 0'].tolist()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(int)
supply_capacity_dict = dict(zip(supply_capacity_df['Unnamed: 0'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0'.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_supplies = transportation_costs_df['Unnamed: 0'].tolist()
cost_customers = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
if set(supplies) != set(cost_supplies):
    raise ValueError('Mismatch between supplies in supply_capacity.csv and transportation_costs.csv.')
if set(customers) != set(cost_customers):
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv.')
cost_dict = {}
for (idx, row) in transportation_costs_df.iterrows():
    s = row['Unnamed: 0']
    for c in customers:
        val = row[c]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for supply {s}, customer {c}: {val}')
        cost_dict[s, c] = cost
m = gp.Model('TransportationOptimization')
x_vars = m.addVars(supplies, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[s, c] * x_vars[s, c] for s in supplies for c in customers)), gp.GRB.MINIMIZE)
for s in supplies:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity_dict[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in supplies)) == demand_dict[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipping Plan (nonzero flows) ---')
    for s in supplies:
        for c in customers:
            val = x_vars[s, c].X
            if val > 1e-06:
                print(f'  Ship {val:.2f} units from {s} to {c} (cost per unit: {cost_dict[s, c]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')