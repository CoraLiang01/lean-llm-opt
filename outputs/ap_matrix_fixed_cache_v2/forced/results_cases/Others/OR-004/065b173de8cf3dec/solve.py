import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
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
supply_capacity_df['Unnamed: 0'] = supply_capacity_df['Unnamed: 0'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
sources = supply_capacity_df['Unnamed: 0'].tolist()
supply_capacity = dict(zip(supply_capacity_df['Unnamed: 0'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0'")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
for c in customers:
    if c not in transportation_costs_df.columns:
        raise KeyError(f"transportation_costs.csv missing column '{c}'")
cost = {}
for idx, row in transportation_costs_df.iterrows():
    s = row['Unnamed: 0']
    for c in customers:
        try:
            cost_val = float(row[c])
        except Exception as e:
            raise ValueError(f"Invalid cost value for source '{s}', customer '{c}': {row[c]}")
        cost[s, c] = cost_val
for s in sources:
    for c in customers:
        if (s, c) not in cost:
            raise KeyError(f"Missing transportation cost for source '{s}', customer '{c}'")
for s in sources:
    if s not in supply_capacity:
        raise KeyError(f"Missing supply capacity for source '{s}'")
for c in customers:
    if c not in demand:
        raise KeyError(f"Missing demand for customer '{c}'")
m = gp.Model('TransportationOptimization')
x_vars = m.addVars(sources, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for s in sources for c in customers)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in sources)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipping Plan (quantities from source to customer) ---')
    for s in sources:
        for c in customers:
            val = x_vars[s, c].X
            if val > 1e-06:
                print(f'  {s} -> {c}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')