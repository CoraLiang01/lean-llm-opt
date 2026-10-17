import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', sep=',', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
demand_df['customer'] = demand_df['customer'].str.strip()
demand_df['demand'] = demand_df['demand'].astype(float)
customers = list(demand_df['customer'])
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
fixed_cost_df['Unnamed: 0'] = fixed_cost_df['Unnamed: 0'].str.strip()
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = list(fixed_cost_df['Unnamed: 0'])
fixed_cost_dict = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in trans_costs_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for suppliers")
trans_costs_df['Unnamed: 0'] = trans_costs_df['Unnamed: 0'].str.strip()
for cust in customers:
    if cust not in trans_costs_df.columns:
        raise KeyError(f"Customer '{cust}' not found as a column in transportation_costs.csv")
trans_cost_dict = {}
for idx, row in trans_costs_df.iterrows():
    supplier = row['Unnamed: 0']
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception as e:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', customer '{cust}': {row[cust]}")
        trans_cost_dict[supplier, cust] = cost
m = gp.Model('UFLP_Superstore')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_term = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
trans_cost_term = gp.quicksum((trans_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_term + trans_cost_term, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        print(f"Supplier {i}: {('OPEN' if y_vars[i].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[i].X))})")
    print('\n--- Supply Plan (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {qty:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')