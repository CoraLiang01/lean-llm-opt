import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = customer_demand_df['customer'].tolist()
demand_c = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
sources = supply_capacity_df['Unnamed: 0'].tolist()
supply_capacity_s = dict(zip(supply_capacity_df['Unnamed: 0'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
cost_s_c = {}
for (idx, row) in transportation_costs_df.iterrows():
    s = row['Unnamed: 0']
    if s not in sources:
        continue
    for c in customers:
        if c not in row:
            raise KeyError(f"Customer '{c}' not found in transportation_costs.csv columns.")
        try:
            cost = float(row[c])
        except Exception as e:
            raise ValueError(f"Invalid cost value for source '{s}', customer '{c}': {row[c]}")
        cost_s_c[s, c] = cost
for s in sources:
    for c in customers:
        if (s, c) not in cost_s_c:
            raise KeyError(f"Missing transportation cost for source '{s}', customer '{c}'.")
m = gp.Model('TransportationProblem')
x_vars = m.addVars(sources, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_s_c[s, c] * x_vars[s, c] for s in sources for c in customers)), gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity_s[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in sources)) == demand_c[c], name=f'demand_{c}')
m.optimize()