import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    fixed_cost_df = read_csv_robust(fixed_cost_path)
    trans_cost_df = read_csv_robust(trans_cost_path)
    warehouses = list(fixed_cost_df['Unnamed: 0'])
    customers = list(demand_df['customer'])
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        if cust in demand:
            demand[cust] += row['demand']
        else:
            demand[cust] = row['demand']
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        wh = row['Unnamed: 0']
        if wh in fixed_cost:
            fixed_cost[wh] += row['fixed_costs']
        else:
            fixed_cost[wh] = row['fixed_costs']
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        wh = row['Unnamed: 0']
        cost[wh] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Customer {cust} not found in transportation_costs.csv columns.')
            cost[wh][cust] = row[cust]
    for wh in warehouses:
        if wh not in fixed_cost:
            raise ValueError(f'Warehouse {wh} missing in fixed_cost data.')
        if wh not in cost:
            raise ValueError(f'Warehouse {wh} missing in transportation cost data.')
        for cust in customers:
            if cust not in cost[wh]:
                raise ValueError(f'Customer {cust} missing in transportation cost data for warehouse {wh}.')
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Customer {cust} missing in demand data.')
    M = sum(demand.values())
    m = gp.Model('Bandcamp_UFLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()