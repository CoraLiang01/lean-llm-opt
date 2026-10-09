import gurobipy as gp
import pandas as pd
import numpy as np
import re

def check_coverage(keys, ref_keys, kind):
    missing = set(keys) - set(ref_keys)
    if missing:
        raise ValueError(f'Missing {kind} in data: {sorted(missing)}')

def solve_transportation_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
    demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    demand_df['customer'] = demand_df['customer'].str.strip()
    demand_df['demand'] = demand_df['demand'].astype(int)
    customers = demand_df['customer'].tolist()
    customer_set = set(customers)
    demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
    supply_df = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
    supply_df['warehouse'] = supply_df['Unnamed: 0'].str.strip()
    supply_df['supply_capacity'] = supply_df['supply_capacity'].astype(int)
    warehouses = supply_df['warehouse'].tolist()
    warehouse_set = set(warehouses)
    supply_dict = dict(zip(supply_df['warehouse'], supply_df['supply_capacity']))
    cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
    cost_df['warehouse'] = cost_df['Unnamed: 0'].str.strip()
    cost_warehouses = cost_df['warehouse'].tolist()
    cost_customers = [col for col in cost_df.columns if col not in ['Unnamed: 0', 'warehouse']]
    check_coverage(warehouses, cost_warehouses, 'warehouses in transportation_costs.csv')
    check_coverage(customers, cost_customers, 'customers in transportation_costs.csv')
    cost_dict = {}
    for (_, row) in cost_df.iterrows():
        w = row['warehouse']
        for c in customers:
            val = row[c]
            try:
                cost = float(val)
            except Exception:
                raise ValueError(f'Invalid cost value for warehouse {w}, customer {c}: {val}')
            cost_dict[w, c] = cost
    m = gp.Model('Transportation')
    m.Params.MIPGap = 0.0001
    index_pairs = [(w, c) for w in warehouses for c in customers]
    quantity_vars = m.addVars(index_pairs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost_dict[w, c] * quantity_vars[w, c] for (w, c) in index_pairs)), gp.GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((quantity_vars[w, c] for w in warehouses)) == demand_dict[c], name=f'demand_{c}')
    for w in warehouses:
        m.addConstr(gp.quicksum((quantity_vars[w, c] for c in customers)) <= supply_dict[w], name=f'supply_{w}')
    m.optimize()
    return m
m = solve_transportation_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')