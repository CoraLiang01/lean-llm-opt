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
branches = demand_df['customer'].str.strip().tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    branch = str(row['customer']).strip()
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for branch {branch}: {row['demand']}")
    demand_dict[branch] = demand_val
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    try:
        fixed_val = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {supplier}: {row['fixed_costs']}")
    fixed_cost_dict[supplier] = fixed_val
transport_cost_dict = {}
for branch in branches:
    if branch not in transport_cost_df.columns:
        raise KeyError(f"Branch '{branch}' not found as a column in transportation_costs.csv")
for (idx, row) in transport_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    if supplier not in suppliers:
        raise KeyError(f"Supplier '{supplier}' in transportation_costs.csv not found in fixed_cost.csv")
    for branch in branches:
        try:
            cost_val = float(row[branch])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {supplier}, branch {branch}: {row[branch]}')
        transport_cost_dict[supplier, branch] = cost_val
if set(suppliers) != set(fixed_cost_dict.keys()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and extracted supplier list.')
if set(branches) != set(demand_dict.keys()):
    raise ValueError('Mismatch between branches in demand.csv and extracted branch list.')
for s in suppliers:
    for b in branches:
        if (s, b) not in transport_cost_dict:
            raise KeyError(f'Missing transportation cost for supplier {s}, branch {b}')
m = gp.Model('UFLP_Superstore')
x_vars = m.addVars(suppliers, branches, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
transport_cost_expr = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in branches))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in branches:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
M = sum(demand_dict.values())
for i in suppliers:
    for j in branches:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()