import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', dtype=str, keep_default_na=False)
customers = demand_df['customer'].str.strip().tolist()
demand_dict = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}") from e
warehouses = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
fixed_cost_dict = {}
for idx, row in fixed_cost_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    try:
        fixed_cost_dict[wh] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed cost for warehouse {wh}: {row['fixed_costs']}") from e
trans_whs = transport_costs_df['Unnamed: 0'].str.strip().tolist()
if set(warehouses) != set(trans_whs):
    raise ValueError(f'Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv: {set(warehouses)} vs {set(trans_whs)}')
trans_custs = [c for c in transport_costs_df.columns if c != 'Unnamed: 0']
if set(customers) != set(trans_custs):
    raise ValueError(f'Mismatch between customers in demand.csv and transportation_costs.csv: {set(customers)} vs {set(trans_custs)}')
transport_cost_dict = {}
for idx, row in transport_costs_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    for cust in customers:
        try:
            val = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for warehouse {wh}, customer {cust}: {row[cust]}') from e
        transport_cost_dict[wh, cust] = val
m = gp.Model('UFLP_Bandcamp')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[wh] * y_vars[wh] for wh in warehouses)) + gp.quicksum((transport_cost_dict[wh, cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)), gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand_dict[cust])
for wh in warehouses:
    for cust in customers:
        m.addConstr(x_vars[wh, cust] <= demand_dict[cust] * y_vars[wh])
m.optimize()