import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', sep=',', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must have columns 'customer' and 'demand'")
demand_df['customer'] = demand_df['customer'].str.strip()
demand_df['demand'] = demand_df['demand'].astype(float)
customers = demand_df['customer'].tolist()
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].str.strip()
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = fixed_cost_df['supplier'].tolist()
fixed_cost_dict = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in trans_costs_df.columns:
    raise KeyError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
trans_costs_df['supplier'] = trans_costs_df['Unnamed: 0'].str.strip()
for cust in customers:
    if cust not in trans_costs_df.columns:
        raise KeyError(f"transportation_costs.csv missing column for customer '{cust}'")
transportation_cost = {}
for idx, row in trans_costs_df.iterrows():
    supplier = row['supplier']
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception as e:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', customer '{cust}': {row[cust]}")
        transportation_cost[supplier, cust] = cost
m = gp.Model('UFLP_Adidas_Supplier_Selection')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
total_transport = gp.quicksum((transportation_cost[i, j] * x_vars[i, j] for i in suppliers for j in customers))
m.setObjective(total_fixed + total_transport, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j])
M = sum(demand_dict.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= M * y_vars[i])
m.optimize()