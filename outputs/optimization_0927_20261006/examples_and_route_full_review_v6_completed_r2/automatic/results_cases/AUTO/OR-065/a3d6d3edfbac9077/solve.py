import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise KeyError("demand.csv must contain columns 'customer' and 'demand'")
customers = demand_df['customer'].str.strip().tolist()
demand_dict = dict(zip(demand_df['customer'].str.strip(), demand_df['demand'].astype(int)))
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise KeyError("fixed_cost.csv must contain columns 'Unnamed: 0' and 'fixed_costs'")
warehouses = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
fixed_cost_dict = dict(zip(fixed_cost_df['Unnamed: 0'].str.strip(), fixed_cost_df['fixed_costs'].astype(float)))
if 'Unnamed: 0' not in transport_cost_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0'")
warehouses_tc = transport_cost_df['Unnamed: 0'].str.strip().tolist()
if set(warehouses) != set(warehouses_tc):
    raise ValueError('Mismatch in warehouse IDs between fixed_cost.csv and transportation_costs.csv')
transport_customer_cols = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(transport_customer_cols):
    raise ValueError('Mismatch in customer IDs between demand.csv and transportation_costs.csv columns')
transport_cost_dict = {}
for (idx, row) in transport_cost_df.iterrows():
    w = str(row['Unnamed: 0']).strip()
    for c in customers:
        val = row[c]
        try:
            transport_cost_dict[w, c] = float(val)
        except Exception:
            raise ValueError(f"Invalid transportation cost for warehouse '{w}', customer '{c}': {val}")
m = gp.Model('UFLP_Bandcamp')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[w] * y_vars[w] for w in warehouses)) + gp.quicksum((transport_cost_dict[w, c] * x_vars[w, c] for w in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[w, c] for w in warehouses)) == demand_dict[c])
for w in warehouses:
    for c in customers:
        m.addConstr(x_vars[w, c] <= demand_dict[c] * y_vars[w])
m.optimize()