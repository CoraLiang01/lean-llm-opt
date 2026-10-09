import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
suppliers = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
customers = demand_df['customer'].str.strip().tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {supplier}: {row['fixed_costs']}")
    fixed_cost_dict[supplier] = fixed_cost
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    customer = str(row['customer']).strip()
    try:
        demand = float(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {customer}: {row['demand']}")
    demand_dict[customer] = demand
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    for customer in customers:
        if customer not in row:
            raise KeyError(f'Customer {customer} not found in transportation_costs.csv columns')
        try:
            cost = float(row[customer])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {supplier}, customer {customer}: {row[customer]}')
        transport_cost_dict[supplier, customer] = cost
if set(suppliers) != set(fixed_cost_dict.keys()):
    raise ValueError('Mismatch between supplier list and fixed_cost_dict keys')
if set(customers) != set(demand_dict.keys()):
    raise ValueError('Mismatch between customer list and demand_dict keys')
for s in suppliers:
    for c in customers:
        if (s, c) not in transport_cost_dict:
            raise KeyError(f'Missing transportation cost for supplier {s}, customer {c}')
m = gp.Model('UFLP_Adidas')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[s] * y_vars[s] for s in suppliers)) + gp.quicksum((transport_cost_dict[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand_dict[c], name=f'demand_{c}')
for s in suppliers:
    for c in customers:
        m.addConstr(x_vars[s, c] <= demand_dict[c] * y_vars[s], name=f'link_{s}_{c}')
m.optimize()