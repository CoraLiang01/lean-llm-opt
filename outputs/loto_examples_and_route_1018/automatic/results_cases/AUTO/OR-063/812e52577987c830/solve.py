import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = demand_df['customer'].str.strip().tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val
warehouses = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed cost for warehouse {wh}: {row['fixed_costs']}")
    fixed_cost_dict[wh] = fc
transport_warehouses = transport_df['Unnamed: 0'].str.strip().tolist()
if set(warehouses) != set(transport_warehouses):
    raise ValueError('Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv')
transport_customers = [col for col in transport_df.columns if col != 'Unnamed: 0']
if set(customers) != set(transport_customers):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
transport_cost_dict = {}
for idx, row in transport_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    for cust in customers:
        try:
            tc = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for warehouse {wh}, customer {cust}: {row[cust]}')
        transport_cost_dict[wh, cust] = tc
m = gp.Model('UFLP_Bandcamp')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in warehouses)) + gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in warehouses for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouses)) == demand_dict[j], name=f'demand_{j}')
for i in warehouses:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demand_dict[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()