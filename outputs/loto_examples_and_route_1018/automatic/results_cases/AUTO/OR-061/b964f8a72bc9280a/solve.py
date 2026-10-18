import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
branches = demand_df['customer'].str.strip().tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    branch = str(row['customer']).strip()
    try:
        demand = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for branch {branch}: {row['demand']}")
    demand_dict[branch] = demand
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    try:
        cost = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {supplier}: {row['fixed_costs']}")
    fixed_cost_dict[supplier] = cost
transport_cost_dict = {}
for idx, row in transport_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    for branch in branches:
        if branch not in row:
            raise KeyError(f'Branch {branch} not found in transportation_costs.csv columns')
        try:
            cost = float(row[branch])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {supplier}, branch {branch}: {row[branch]}')
        transport_cost_dict[supplier, branch] = cost
if set(suppliers) != set(fixed_cost_dict.keys()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and extracted supplier list')
if set(branches) != set(demand_dict.keys()):
    raise ValueError('Mismatch between branches in demand.csv and extracted branch list')
for s in suppliers:
    for c in branches:
        if (s, c) not in transport_cost_dict:
            raise ValueError(f'Missing transportation cost for supplier {s}, branch {c}')
M = sum((demand_dict[c] for c in branches))
m = gp.Model('UFLP_Superstore')
x_vars = m.addVars(suppliers, branches, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[s] * y_vars[s] for s in suppliers)) + gp.quicksum((transport_cost_dict[s, c] * x_vars[s, c] for s in suppliers for c in branches)), gp.GRB.MINIMIZE)
for c in branches:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand_dict[c], name=f'demand_{c}')
for s in suppliers:
    for c in branches:
        m.addConstr(x_vars[s, c] <= M * y_vars[s], name=f'link_{s}_{c}')
m.optimize()