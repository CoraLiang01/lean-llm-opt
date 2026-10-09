import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', dtype=str, keep_default_na=False)
customer_ids = demand_df['customer'].astype(str).str.strip().tolist()
supplier_ids = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
transport_supplier_ids = transport_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
if set(supplier_ids) != set(transport_supplier_ids):
    raise ValueError(f'Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv: {set(supplier_ids)} vs {set(transport_supplier_ids)}')
transport_customer_ids = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(customer_ids) != set(transport_customer_ids):
    raise ValueError(f'Mismatch between customers in demand.csv and transportation_costs.csv: {set(customer_ids)} vs {set(transport_customer_ids)}')
demand = {}
for (idx, row) in demand_df.iterrows():
    cust = str(row['customer']).strip()
    try:
        demand[cust] = int(row['demand'])
    except Exception:
        raise ValueError(f"Non-integer demand for customer {cust}: {row['demand']}")
fixed_cost = {}
for (idx, row) in fixed_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost[sup] = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Non-numeric fixed_cost for supplier {sup}: {row['fixed_costs']}")
transport_cost = {}
for (idx, row) in transport_cost_df.iterrows():
    sup = str(row['Unnamed: 0']).strip()
    for cust in customer_ids:
        try:
            val = float(row[cust])
        except Exception:
            raise ValueError(f'Non-numeric transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        transport_cost[sup, cust] = val
supplier_set = supplier_ids
customer_set = customer_ids
x_keys = [(i, j) for i in supplier_set for j in customer_set]

def solve_uflp(supplier_set, customer_set, demand, fixed_cost, transport_cost):
    m = gp.Model('UFLP')
    y_vars = m.addVars(supplier_set, vtype=gp.GRB.BINARY, name='')
    x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in supplier_set)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in supplier_set for j in customer_set)), gp.GRB.MINIMIZE)
    for j in customer_set:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in supplier_set)) == demand[j], name=f'demand_{j}')
    for i in supplier_set:
        for j in customer_set:
            m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name=f'link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_uflp(supplier_set, customer_set, demand, fixed_cost, transport_cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')