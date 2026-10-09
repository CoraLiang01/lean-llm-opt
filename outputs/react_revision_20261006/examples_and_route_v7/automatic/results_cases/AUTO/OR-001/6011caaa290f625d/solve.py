import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if not {'customer', 'demand'}.issubset(customer_demand_df.columns):
    raise ValueError("customer_demand.csv must contain columns 'customer' and 'demand'")
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = customer_demand_df['customer'].tolist()
demand = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if not {'Unnamed: 0', 'supply_capacity'}.issubset(supply_capacity_df.columns):
    raise ValueError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'")
supply_capacity_df['Unnamed: 0'] = supply_capacity_df['Unnamed: 0'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
sources = supply_capacity_df['Unnamed: 0'].tolist()
supply_capacity = dict(zip(supply_capacity_df['Unnamed: 0'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise ValueError("transportation_costs.csv must contain column 'Unnamed: 0' for source IDs")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_sources = transportation_costs_df['Unnamed: 0'].tolist()
cost_customers = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
if set(sources) != set(cost_sources):
    raise ValueError('Mismatch between sources in supply_capacity.csv and transportation_costs.csv')
if set(customers) != set(cost_customers):
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv')
cost = {}
for (idx, row) in transportation_costs_df.iterrows():
    s = row['Unnamed: 0']
    for c in customers:
        val = row[c]
        try:
            cost_val = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for source {s}, customer {c}: {val}')
        cost[s, c] = cost_val
for s in sources:
    if s not in supply_capacity:
        raise ValueError(f'Supply capacity missing for source {s}')
for c in customers:
    if c not in demand:
        raise ValueError(f'Demand missing for customer {c}')
for s in sources:
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Transportation cost missing for source {s}, customer {c}')
decision_keys = [(s, c) for s in sources for c in customers]
m = gp.Model('Amazon_Transportation')
m.Params.MIPGap = 0.0001
quantity_vars = m.addVars(decision_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * quantity_vars[s, c] for (s, c) in decision_keys)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((quantity_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((quantity_vars[s, c] for s in sources)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for (s, c) in decision_keys:
        var = quantity_vars[s, c]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')