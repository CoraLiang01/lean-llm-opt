import gurobipy as gp
import pandas as pd
import numpy as np

def solve_uflp():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', sep=',')
    fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', sep=',')
    trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', sep=',')
    suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    customers = demand_df['customer'].astype(str).str.strip().tolist()
    trans_suppliers = trans_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    trans_customers = [c for c in trans_cost_df.columns if c != 'Unnamed: 0']
    if set(suppliers) != set(trans_suppliers):
        raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(trans_suppliers)}')
    if set(customers) != set(trans_customers):
        raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers)} vs {set(trans_customers)}')
    fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
    demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
    trans_costs = {}
    for idx, row in trans_cost_df.iterrows():
        supplier = str(row['Unnamed: 0']).strip()
        trans_costs[supplier] = {}
        for customer in customers:
            trans_costs[supplier][customer] = float(row[customer])
    m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
    y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
    x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((trans_costs[i][j] * x[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((x[i, j] <= demand[j] * y[i] for i in suppliers for j in customers), name='')
    m.optimize()
    return m
m = solve_uflp()