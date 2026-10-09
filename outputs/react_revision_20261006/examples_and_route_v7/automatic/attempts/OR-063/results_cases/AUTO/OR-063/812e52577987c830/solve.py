import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if set(['customer', 'demand']) - set(demand_df.columns):
    raise ValueError('Missing required columns in demand.csv')
demand_df['customer'] = demand_df['customer'].str.strip()
customers = demand_df['customer'].tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_dict[cust] = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
if set(['Unnamed: 0', 'fixed_costs']) - set(fixed_cost_df.columns):
    raise ValueError('Missing required columns in fixed_cost.csv')
fixed_cost_df['Unnamed: 0'] = fixed_cost_df['Unnamed: 0'].str.strip()
warehouses = fixed_cost_df['Unnamed: 0'].tolist()
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    try:
        fixed_cost_dict[wh] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed_costs value for warehouse {wh}: {row['fixed_costs']}")
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transport_df.columns:
    raise ValueError('Missing Unnamed: 0 column in transportation_costs.csv')
transport_df['Unnamed: 0'] = transport_df['Unnamed: 0'].str.strip()
if set(warehouses) - set(transport_df['Unnamed: 0']):
    raise ValueError('Some warehouses in fixed_cost.csv are missing in transportation_costs.csv')
if set(customers) - set(transport_df.columns):
    raise ValueError('Some customers in demand.csv are missing in transportation_costs.csv')
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    for cust in customers:
        try:
            transport_cost_dict[wh, cust] = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for warehouse {wh}, customer {cust}: {row[cust]}')
for wh in warehouses:
    if wh not in fixed_cost_dict:
        raise ValueError(f'Warehouse {wh} missing fixed cost')
    for cust in customers:
        if (wh, cust) not in transport_cost_dict:
            raise ValueError(f'Missing transportation cost for warehouse {wh}, customer {cust}')
x_keys = [(wh, cust) for wh in warehouses for cust in customers]
y_keys = warehouses

def solve_problem():
    m = gp.Model('UFLP_Bandcamp')
    x_vars = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    obj = gp.quicksum((fixed_cost_dict[wh] * y_vars[wh] for wh in warehouses)) + gp.quicksum((transport_cost_dict[wh, cust] * x_vars[wh, cust] for (wh, cust) in x_keys))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    for cust in customers:
        m.addConstr(gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand_dict[cust], name=f'demand_{cust}')
    for wh in warehouses:
        for cust in customers:
            m.addConstr(x_vars[wh, cust] <= demand_dict[cust] * y_vars[wh], name=f'link_{wh}_{cust}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')