import gurobipy as gp
import pandas as pd
import numpy as np

def solve_transportation_problem():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv', dtype=str, keep_default_na=False)
    supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv', dtype=str, keep_default_na=False)
    cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv', dtype=str, keep_default_na=False)
    S = supply_df['Unnamed: 0'].tolist()
    C = demand_df['customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        c = str(row['customer']).strip()
        try:
            d = int(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {c}: {row['demand']}")
        demand[c] = d
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        s = str(row['Unnamed: 0']).strip()
        try:
            cap = int(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for source {s}: {row['supply_capacity']}")
        supply_capacity[s] = cap
    cost = {}
    cost_df_indexed = cost_df.set_index('Unnamed: 0')
    for s in S:
        if s not in cost_df_indexed.index:
            raise KeyError(f'Source {s} not found in transportation_costs.csv')
        for c in C:
            if c not in cost_df_indexed.columns:
                raise KeyError(f'Customer {c} not found as column in transportation_costs.csv')
            try:
                val = float(cost_df_indexed.loc[s, c])
            except Exception:
                raise ValueError(f'Invalid cost value for source {s}, customer {c}: {cost_df_indexed.loc[s, c]}')
            cost[s, c] = val
    if set(demand.keys()) != set(C):
        raise ValueError('Mismatch between customer_demand.csv and transportation_costs.csv columns')
    if set(supply_capacity.keys()) != set(S):
        raise ValueError('Mismatch between supply_capacity.csv and transportation_costs.csv rows')
    for s in S:
        for c in C:
            if (s, c) not in cost:
                raise ValueError(f'Missing cost coefficient for ({s}, {c})')
    m = gp.Model('Transportation')
    m.Params.MIPGap = 0.0001
    shipment_vars = m.addVars([(s, c) for s in S for c in C], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[s, c] * shipment_vars[s, c] for s in S for c in C)), gp.GRB.MINIMIZE)
    for c in C:
        m.addConstr(gp.quicksum((shipment_vars[s, c] for s in S)) == demand[c], name='')
    for s in S:
        m.addConstr(gp.quicksum((shipment_vars[s, c] for c in C)) <= supply_capacity[s], name='')
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for ((s, c), var) in shipment_vars.items():
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_transportation_problem()