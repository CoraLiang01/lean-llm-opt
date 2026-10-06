import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    df_demand = read_csv_robust(customer_demand_path)
    df_supply = read_csv_robust(supply_capacity_path)
    df_cost = read_csv_robust(transportation_costs_path)
    I = df_supply['Unnamed: 0'].tolist()
    J = df_demand['customer'].tolist()
    demand = {}
    for (_, row) in df_demand.iterrows():
        j = row['customer']
        if j in demand:
            demand[j] += row['demand']
        else:
            demand[j] = row['demand']
    supply_capacity = {}
    for (_, row) in df_supply.iterrows():
        i = row['Unnamed: 0']
        if i in supply_capacity:
            supply_capacity[i] += row['supply_capacity']
        else:
            supply_capacity[i] = row['supply_capacity']
    cost = {}
    for (_, row) in df_cost.iterrows():
        i = row['Unnamed: 0']
        cost[i] = {}
        for j in J:
            if j not in row:
                raise ValueError(f'Missing cost column for customer {j} in transportation_costs.csv')
            cost[i][j] = row[j]
    for j in J:
        if j not in demand:
            raise ValueError(f'Demand data missing for customer {j}')
    for i in I:
        if i not in supply_capacity:
            raise ValueError(f'Supply capacity data missing for supplier {i}')
        if i not in cost:
            raise ValueError(f'Cost data missing for supplier {i}')
        for j in J:
            if j not in cost[i]:
                raise ValueError(f'Cost data missing for supplier {i}, customer {j}')
    m = gp.Model('Amazon_Distribution_Transportation')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in I for j in J]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= supply_capacity[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()