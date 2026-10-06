import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    supply_df = read_csv_robust(supply_path)
    cost_df = read_csv_robust(cost_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError('customer_demand.csv must have columns: customer, demand')
    customers = demand_df['customer'].astype(str).tolist()
    demand = demand_df.set_index('customer')['demand'].to_dict()
    plant_col = supply_df.columns[0]
    if 'supply_capacity' not in supply_df.columns:
        raise ValueError('supply_capacity.csv must have column: supply_capacity')
    plants = supply_df[plant_col].astype(str).tolist()
    supply_capacity = supply_df.set_index(plant_col)['supply_capacity'].to_dict()
    cost_plant_col = cost_df.columns[0]
    cost_customers = [col for col in cost_df.columns if col != cost_plant_col]
    if set(cost_customers) != set(customers):
        raise ValueError('Mismatch between customers in cost and demand files.')
    if set(cost_df[cost_plant_col].astype(str)) != set(plants):
        raise ValueError('Mismatch between plants in cost and supply files.')
    cost = {}
    for (_, row) in cost_df.iterrows():
        plant = str(row[cost_plant_col])
        cost[plant] = {}
        for cust in cost_customers:
            cost[plant][cust] = row[cust]
    for i in plants:
        if i not in cost:
            raise ValueError(f'Missing cost row for plant {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for plant {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in plants:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for plant {i}')
    m = gp.Model('BrewCo_Transportation')
    m.setParam('MIPGap', 0.0001)
    keys = [(i, j) for i in plants for j in customers]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in plants)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in plants), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()