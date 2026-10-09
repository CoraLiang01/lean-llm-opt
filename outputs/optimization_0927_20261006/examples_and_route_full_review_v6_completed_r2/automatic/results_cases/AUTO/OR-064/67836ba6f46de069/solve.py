import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
customers = demand_df['customer'].str.strip().tolist()
transport_suppliers = transport_df['Unnamed: 0'].str.strip().tolist()
if set(suppliers) != set(transport_suppliers):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
transport_customers = [c for c in transport_df.columns if c != 'Unnamed: 0']
if set(customers) != set(transport_customers):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
fixed_costs = {}
for (idx, row) in fixed_cost_df.iterrows():
    sid = row['Unnamed: 0'].strip()
    try:
        fixed_costs[sid] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {sid}: {row['fixed_costs']}") from e
demands = {}
for (idx, row) in demand_df.iterrows():
    cid = row['customer'].strip()
    try:
        demands[cid] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cid}: {row['demand']}") from e
transport_costs = {}
for (idx, row) in transport_df.iterrows():
    sid = row['Unnamed: 0'].strip()
    for cid in customers:
        try:
            val = row[cid]
            transport_costs[sid, cid] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sid}, customer {cid}: {row[cid]}') from e
m = gp.Model('UFLP')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transport_costs[i, j] * x_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demands[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demands[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()