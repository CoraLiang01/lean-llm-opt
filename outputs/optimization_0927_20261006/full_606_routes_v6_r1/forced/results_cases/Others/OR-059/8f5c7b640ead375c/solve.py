import gurobipy as gp
import pandas as pd
import numpy as np
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
supplier_ids = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_costs = fixed_cost_df.set_index('Unnamed: 0')['fixed_costs'].astype(float).to_dict()
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise ValueError("demand.csv must have columns 'customer' and 'demand'")
dealership_ids = demand_df['customer'].astype(str).str.strip().tolist()
demands = demand_df.set_index('customer')['demand'].astype(float).to_dict()
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in trans_costs_df.columns:
    raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
for d in dealership_ids:
    if d not in trans_costs_df.columns:
        raise ValueError(f"Dealership '{d}' not found as a column in transportation_costs.csv")
trans_costs = {}
for (_, row) in trans_costs_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    if supplier not in supplier_ids:
        continue
    for d in dealership_ids:
        try:
            cost = float(row[d])
        except Exception:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', dealership '{d}'")
        trans_costs[supplier, d] = cost
for i in supplier_ids:
    for j in dealership_ids:
        if (i, j) not in trans_costs:
            raise ValueError(f"Missing transportation cost for supplier '{i}', dealership '{j}'")
for i in supplier_ids:
    if i not in fixed_costs:
        raise ValueError(f"Missing fixed cost for supplier '{i}'")
for j in dealership_ids:
    if j not in demands:
        raise ValueError(f"Missing demand for dealership '{j}'")
I = supplier_ids
J = dealership_ids
M = sum((demands[j] for j in J))
m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
x_vars = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y_vars[i] for i in I)) + gp.quicksum((trans_costs[i, j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == demands[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x_vars[i, j] <= M * y_vars[i], name=f'link_{i}_{j}')
m.optimize()