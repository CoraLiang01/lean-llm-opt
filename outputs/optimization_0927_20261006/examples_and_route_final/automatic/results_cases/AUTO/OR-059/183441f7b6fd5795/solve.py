import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', dtype=str, keep_default_na=False)
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', dtype=str, keep_default_na=False)
supplier_ids = list(fixed_cost_df['Unnamed: 0'].astype(str))
trans_supplier_ids = list(trans_cost_df['Unnamed: 0'].astype(str))
if set(supplier_ids) != set(trans_supplier_ids):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
customer_ids = list(demand_df['customer'].astype(str))
trans_customer_ids = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
if set(customer_ids) != set(trans_customer_ids):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
fixed_costs = {}
for (idx, row) in fixed_cost_df.iterrows():
    sid = str(row['Unnamed: 0'])
    try:
        fixed_costs[sid] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs for supplier {sid}: {row['fixed_costs']}") from e
demands = {}
for (idx, row) in demand_df.iterrows():
    cid = str(row['customer'])
    try:
        demands[cid] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand for customer {cid}: {row['demand']}") from e
trans_costs = {}
for (idx, row) in trans_cost_df.iterrows():
    sid = str(row['Unnamed: 0'])
    for cid in customer_ids:
        try:
            trans_costs[sid, cid] = float(row[cid])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sid}, customer {cid}: {row[cid]}') from e
m = gp.Model('Colorado_Motor_Vehicle_UFLP')
y_vars = m.addVars(supplier_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(supplier_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
for cid in customer_ids:
    m.addConstr(gp.quicksum((x_vars[sid, cid] for sid in supplier_ids)) == demands[cid], name=f'demand_{cid}')
for sid in supplier_ids:
    for cid in customer_ids:
        m.addConstr(x_vars[sid, cid] <= demands[cid] * y_vars[sid], name=f'link_{sid}_{cid}')
fixed_cost_term = gp.quicksum((fixed_costs[sid] * y_vars[sid] for sid in supplier_ids))
trans_cost_term = gp.quicksum((trans_costs[sid, cid] * x_vars[sid, cid] for sid in supplier_ids for cid in customer_ids))
m.setObjective(fixed_cost_term + trans_cost_term, gp.GRB.MINIMIZE)
m.optimize()