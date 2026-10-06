import gurobipy as gp
import pandas as pd
import numpy as np

def solve_uflp():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv', sep=',')
    fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv', sep=',')
    trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', sep=',')
    warehouses = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    warehouses_tc = trans_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
    if set(warehouses) != set(warehouses_tc):
        raise ValueError('Mismatch in warehouse identifiers between fixed_cost.csv and transportation_costs.csv')
    warehouses = sorted(warehouses)
    customers = demand_df['customer'].astype(str).str.strip().tolist()
    customers_tc = [c for c in trans_cost_df.columns if c != 'Unnamed: 0']
    if set(customers) != set(customers_tc):
        raise ValueError('Mismatch in customer identifiers between demand.csv and transportation_costs.csv')
    customers = sorted(customers)
    fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs'].astype(float)))
    demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand'].astype(float)))
    trans_cost = {}
    for (idx, row) in trans_cost_df.iterrows():
        w = str(row['Unnamed: 0']).strip()
        for c in customers:
            trans_cost[w, c] = float(row[c])
    for w in warehouses:
        if w not in fixed_costs:
            raise ValueError(f'Missing fixed cost for warehouse {w}')
        for c in customers:
            if (w, c) not in trans_cost:
                raise ValueError(f'Missing transportation cost for warehouse {w}, customer {c}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    m = gp.Model('UFLP_Bandcamp')
    m.Params.MIPGap = 0.0001
    x_keys = [(w, c) for w in warehouses for c in customers]
    x = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
    obj = gp.quicksum((fixed_costs[w] * y[w] for w in warehouses)) + gp.quicksum((trans_cost[w, c] * x[w, c] for w in warehouses for c in customers))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((x[w, c] for w in warehouses)) == demand[c], name=f'demand_{c}')
    for w in warehouses:
        for c in customers:
            m.addConstr(x[w, c] <= demand[c] * y[w], name='link_{}_{}'.format(w, c))
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for w in warehouses:
            print(f'{y[w].VarName} {y[w].X}')
        for (w, c) in x_keys:
            print(f'{x[w, c].VarName} {x[w, c].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_uflp()