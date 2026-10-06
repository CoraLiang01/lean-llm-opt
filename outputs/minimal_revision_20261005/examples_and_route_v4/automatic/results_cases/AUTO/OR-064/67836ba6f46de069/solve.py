import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'

def solve_uflp():
    demand_df = pd.read_csv(demand_path, sep=',')
    customers = demand_df['customer'].astype(str).tolist()
    demand = dict(zip(demand_df['customer'].astype(str), demand_df['demand'].astype(float)))
    fixed_df = pd.read_csv(fixed_cost_path, sep=',')
    suppliers = fixed_df['Unnamed: 0'].astype(str).tolist()
    fixed_costs = dict(zip(fixed_df['Unnamed: 0'].astype(str), fixed_df['fixed_costs'].astype(float)))
    trans_df = pd.read_csv(transport_cost_path, sep=',')
    trans_df['Unnamed: 0'] = trans_df['Unnamed: 0'].astype(str)
    if set(suppliers) != set(trans_df['Unnamed: 0']):
        raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
    if set(customers) - set(trans_df.columns[1:]):
        raise ValueError('Some customers in demand.csv are missing in transportation_costs.csv columns')
    transportation_costs = {}
    for (_, row) in trans_df.iterrows():
        i = str(row['Unnamed: 0'])
        for j in customers:
            transportation_costs[i, j] = float(row[j])
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in suppliers for j in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    obj = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_costs[i, j] * x[i, j] for i in suppliers for j in customers))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
    M = sum(demand.values())
    for i in suppliers:
        for j in customers:
            m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_uflp()