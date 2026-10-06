import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
    try:
        demand_df = pd.read_csv(demand_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            demand_df = pd.read_csv(demand_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                demand_df = pd.read_csv(demand_path, encoding='gbk')
            except UnicodeDecodeError:
                demand_df = pd.read_csv(demand_path, encoding='latin-1')
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError('Missing required columns in demand.csv')
    customers = demand_df['customer'].astype(str).tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = str(row['customer'])
        if key in demand:
            demand[key] += float(row['demand'])
        else:
            demand[key] = float(row['demand'])
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
    try:
        fixed_cost_df = pd.read_csv(fixed_cost_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            fixed_cost_df = pd.read_csv(fixed_cost_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                fixed_cost_df = pd.read_csv(fixed_cost_path, encoding='gbk')
            except UnicodeDecodeError:
                fixed_cost_df = pd.read_csv(fixed_cost_path, encoding='latin-1')
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError('Missing required columns in fixed_cost.csv')
    warehouses = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        key = str(row['Unnamed: 0'])
        if key in fixed_cost:
            fixed_cost[key] += float(row['fixed_costs'])
        else:
            fixed_cost[key] = float(row['fixed_costs'])
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
    try:
        trans_cost_df = pd.read_csv(trans_cost_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            trans_cost_df = pd.read_csv(trans_cost_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                trans_cost_df = pd.read_csv(trans_cost_path, encoding='gbk')
            except UnicodeDecodeError:
                trans_cost_df = pd.read_csv(trans_cost_path, encoding='latin-1')
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError('Missing required columns in transportation_costs.csv')
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        i = str(row['Unnamed: 0'])
        cost[i] = {}
        for j in customers:
            if j not in trans_cost_df.columns:
                raise ValueError(f'Customer {j} not found in transportation_costs.csv columns')
            cost[i][j] = float(row[j])
    for i in warehouses:
        if i not in fixed_cost:
            raise ValueError(f'Warehouse {i} missing fixed cost')
        if i not in cost:
            raise ValueError(f'Warehouse {i} missing in transportation costs')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost for warehouse {i}, customer {j} missing')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Customer {j} missing demand')
    M = sum((demand[j] for j in customers))
    m = gp.Model('Bandcamp_Warehouse_Selection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()