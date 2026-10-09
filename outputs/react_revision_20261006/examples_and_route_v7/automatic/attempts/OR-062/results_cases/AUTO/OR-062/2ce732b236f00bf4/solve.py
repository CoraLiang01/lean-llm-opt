import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
demand_df['Customer'] = demand_df['Customer'].str.strip()
demand_df['demand'] = demand_df['demand'].astype(float)
customers = list(demand_df['Customer'])
demand_dict = dict(zip(demand_df['Customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
fixed_cost_df['Supplier'] = fixed_cost_df['Unnamed: 0'].str.strip()
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = list(fixed_cost_df['Supplier'])
fixed_cost_dict = dict(zip(fixed_cost_df['Supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
trans_cost_df['Supplier'] = trans_cost_df['Unnamed: 0'].str.strip()
trans_cost_customer_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0' and col != 'Supplier']
trans_cost_customer_cols_norm = [col.strip() for col in trans_cost_customer_cols]
col_norm_map = dict(zip(trans_cost_customer_cols, trans_cost_customer_cols_norm))
trans_cost_df = trans_cost_df.rename(columns=col_norm_map)
transport_cost = {}
for (_, row) in trans_cost_df.iterrows():
    supplier = row['Supplier']
    for cust_col in trans_cost_customer_cols_norm:
        val = row[cust_col]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f"Missing or invalid transportation cost for supplier '{supplier}', customer '{cust_col}'")
        transport_cost[supplier, cust_col] = cost
for s in suppliers:
    if s not in fixed_cost_dict:
        raise ValueError(f"Supplier '{s}' missing from fixed_cost.csv")
    for c in customers:
        if (s, c) not in transport_cost:
            raise ValueError(f"Missing transportation cost for supplier '{s}', customer '{c}'")
for c in customers:
    if c not in demand_dict:
        raise ValueError(f"Customer '{c}' missing from demand.csv")
M = sum((demand_dict[c] for c in customers))
m = gp.Model('UFLP_Iowa_Liquor')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[s] * y_vars[s] for s in suppliers))
trans_cost_expr = gp.quicksum((transport_cost[s, c] * x_vars[s, c] for s in suppliers for c in customers))
m.setObjective(fixed_cost_expr + trans_cost_expr, gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand_dict[c], name=f'demand_{c}')
for s in suppliers:
    for c in customers:
        m.addConstr(x_vars[s, c] <= M * y_vars[s], name=f'link_{s}_{c}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')