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
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
branches = demand_df['customer'].astype(str).str.strip().tolist()
branch_set = set(branches)
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    branch = str(row['customer']).strip()
    try:
        demand = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for branch {branch}: {row['demand']}")
    demand_dict[branch] = demand
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
supplier_set = set(suppliers)
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for supplier {supplier}: {row['fixed_costs']}")
    fixed_cost_dict[supplier] = fixed_cost
if 'Unnamed: 0' not in transport_cost_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
transport_cost_df = transport_cost_df.set_index(transport_cost_df['Unnamed: 0'].astype(str).str.strip())
missing_suppliers = supplier_set - set(transport_cost_df.index)
if missing_suppliers:
    raise KeyError(f'Suppliers {missing_suppliers} in fixed_cost.csv missing from transportation_costs.csv')
missing_branches = branch_set - set([c for c in transport_cost_df.columns if c != 'Unnamed: 0'])
if missing_branches:
    raise KeyError(f'Branches {missing_branches} in demand.csv missing from transportation_costs.csv')
transport_cost_dict = {}
for supplier in suppliers:
    for branch in branches:
        try:
            cost = float(transport_cost_df.loc[supplier, branch])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {supplier}, branch {branch}: {transport_cost_df.loc[supplier, branch]}')
        transport_cost_dict[supplier, branch] = cost
I = suppliers
J = branches
M = sum((demand_dict[j] for j in J))
m = gp.Model('UFLP_Superstore')
x_vars = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
total_fixed_cost = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in I))
total_transport_cost = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in I for j in J))
m.setObjective(total_fixed_cost + total_transport_cost, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == demand_dict[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()