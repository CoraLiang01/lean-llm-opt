import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_uflp():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', sep=',')
    fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', sep=',')
    trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', sep=',')
    customers = demand_df['customer'].astype(str).str.strip().tolist()
    demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
    suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
    trans_suppliers = trans_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    if set(suppliers) != set(trans_suppliers):
        raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers) ^ set(trans_suppliers)}')
    trans_customer_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    if set(customers) != set(trans_customer_cols):
        raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers) ^ set(trans_customer_cols)}')
    transportation_costs = {}
    for (idx, row) in trans_cost_df.iterrows():
        supplier = str(row['Unnamed: 0']).strip()
        for customer in customers:
            transportation_costs[supplier, customer] = float(row[customer])
    I = suppliers
    J = customers
    M = sum((demand[j] for j in J))
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
    total_fixed = gp.quicksum((fixed_costs[i] * y[i] for i in I))
    total_trans = gp.quicksum((transportation_costs[i, j] * x[i, j] for i in I for j in J))
    m.setObjective(total_fixed + total_trans, gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == demand[j] for j in J), name='')
    m.addConstrs((x[i, j] <= M * y[i] for i in I for j in J), name='')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_uflp()