import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', dtype=str, keep_default_na=False)
if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
    raise ValueError("demand.csv must have columns 'customer' and 'demand'")
demand_df['customer'] = demand_df['customer'].str.strip()
demand_df['demand'] = demand_df['demand'].astype(np.int64)
customers = demand_df['customer'].tolist()
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
    raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].str.strip()
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = fixed_cost_df['supplier'].tolist()
fixed_cost_dict = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in trans_costs_df.columns:
    raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
trans_costs_df['supplier'] = trans_costs_df['Unnamed: 0'].str.strip()
for cust in customers:
    if cust not in trans_costs_df.columns:
        raise ValueError(f"transportation_costs.csv missing column for customer '{cust}'")
for cust in customers:
    trans_costs_df[cust] = trans_costs_df[cust].astype(float)
trans_cost_dict = {}
for (_, row) in trans_costs_df.iterrows():
    supplier = row['supplier']
    for cust in customers:
        trans_cost_dict[supplier, cust] = row[cust]
if set(suppliers) != set(trans_costs_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if not set(customers).issubset(set(trans_costs_df.columns)):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv columns')
m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
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