import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
    supply_path = 'Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
    cost_path = 'Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
    df_demand = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    df_supply = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
    df_cost = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
    demand = {}
    cols = df_demand.columns.tolist()
    (key_col, val_col) = (cols[0], cols[1])
    for (_, row) in df_demand.iterrows():
        key = row[key_col].strip()
        val_str = row[val_col].strip()
        if val_str == '':
            raise ValueError(f'Blank demand value for {key}')
        demand[key] = float(val_str)
    supply = {}
    cols = df_supply.columns.tolist()
    (key_col, val_col) = (cols[0], cols[1])
    for (_, row) in df_supply.iterrows():
        key = row[key_col].strip()
        val_str = row[val_col].strip()
        if val_str == '':
            raise ValueError(f'Blank supply value for {key}')
        supply[key] = float(val_str)
    cost = {}
    cols = df_cost.columns.tolist()
    key_col = cols[0]
    for (_, row) in df_cost.iterrows():
        plant = row[key_col].strip()
        cost[plant] = {}
        for cust in cols[1:]:
            val_str = row[cust].strip()
            if val_str == '':
                raise ValueError(f'Blank cost value for {plant},{cust}')
            cost[plant][cust] = float(val_str)
    I = ['S1', 'S2', 'S3', 'S4']
    J = ['C1', 'C2', 'C3', 'C4']
    if not set(I).issubset(supply.keys()):
        missing = set(I) - set(supply.keys())
        raise ValueError(f'Missing supply data for plants: {missing}')
    if not set(J).issubset(demand.keys()):
        missing = set(J) - set(demand.keys())
        raise ValueError(f'Missing demand data for customers: {missing}')
    for i in I:
        if i not in cost:
            raise ValueError(f'Missing cost data for plant {i}')
        if not set(J).issubset(cost[i].keys()):
            missing = set(J) - set(cost[i].keys())
            raise ValueError(f'Missing cost data for plant {i} to customers: {missing}')
    m = gurobipy.Model('TP9')
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in J)) <= supply[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()