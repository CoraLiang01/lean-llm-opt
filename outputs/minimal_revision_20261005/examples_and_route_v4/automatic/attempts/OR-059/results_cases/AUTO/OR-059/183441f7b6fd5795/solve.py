import gurobipy as gp
import pandas as pd
import numpy as np

def solve_uflp():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv'
    demand_df = pd.read_csv(demand_path, sep=',')
    fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
    trans_cost_df = pd.read_csv(trans_cost_path, sep=',')
    suppliers_fc = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    suppliers_tc = trans_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    if set(suppliers_fc) != set(suppliers_tc):
        raise ValueError('Supplier sets in fixed_cost.csv and transportation_costs.csv do not match.')
    suppliers = suppliers_fc
    customers_demand = demand_df['customer'].astype(str).str.strip().tolist()
    customers_tc = [c for c in trans_cost_df.columns if c.startswith('C')]
    if set(customers_demand) != set(customers_tc):
        raise ValueError('Customer sets in demand.csv and transportation_costs.csv do not match.')
    customers = customers_demand
    fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
    demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
    trans_cost = {}
    for (idx, row) in trans_cost_df.iterrows():
        s = str(row['Unnamed: 0']).strip()
        for c in customers:
            if c not in trans_cost_df.columns:
                raise ValueError(f'Customer {c} not found in transportation_costs.csv columns.')
            cost = row[c]
            trans_cost[s, c] = float(cost)
    for s in suppliers:
        if s not in fixed_costs:
            raise ValueError(f'Missing fixed cost for supplier {s}')
        for c in customers:
            if (s, c) not in trans_cost:
                raise ValueError(f'Missing transportation cost for supplier {s}, customer {c}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
    m.Params.MIPGap = 0.0001
    x = m.addVars([(s, c) for s in suppliers for c in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
    obj = gp.quicksum((fixed_costs[s] * y[s] for s in suppliers)) + gp.quicksum((trans_cost[s, c] * x[s, c] for s in suppliers for c in customers))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
    for s in suppliers:
        for c in customers:
            m.addConstr(x[s, c] <= demand[c] * y[s], name=f'link_{s}_{c}')
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_uflp()