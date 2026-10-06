import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
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
    if not {'customer', 'demand'}.issubset(demand_df.columns):
        raise ValueError('Missing required columns in demand.csv')
    customers = demand_df['customer'].astype(str).tolist()
    demand = demand_df.groupby('customer')['demand'].sum().to_dict()
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
    try:
        fixed_df = pd.read_csv(fixed_cost_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            fixed_df = pd.read_csv(fixed_cost_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                fixed_df = pd.read_csv(fixed_cost_path, encoding='gbk')
            except UnicodeDecodeError:
                fixed_df = pd.read_csv(fixed_cost_path, encoding='latin-1')
    if not {'Unnamed: 0', 'fixed_costs'}.issubset(fixed_df.columns):
        raise ValueError('Missing required columns in fixed_cost.csv')
    suppliers = fixed_df['Unnamed: 0'].astype(str).tolist()
    fixed_cost = fixed_df.set_index('Unnamed: 0')['fixed_costs'].to_dict()
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
    try:
        trans_df = pd.read_csv(trans_cost_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            trans_df = pd.read_csv(trans_cost_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                trans_df = pd.read_csv(trans_cost_path, encoding='gbk')
            except UnicodeDecodeError:
                trans_df = pd.read_csv(trans_cost_path, encoding='latin-1')
    if 'Unnamed: 0' not in trans_df.columns:
        raise ValueError("Missing 'Unnamed: 0' column in transportation_costs.csv")
    cost_customer_cols = [col for col in trans_df.columns if col != 'Unnamed: 0']
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Missing transportation cost columns for customers: {missing_customers}')
    cost = {}
    for (_, row) in trans_df.iterrows():
        supplier = str(row['Unnamed: 0'])
        cost[supplier] = {}
        for customer in customers:
            cost[supplier][customer] = row[customer]
    missing_suppliers = set(suppliers) - set(cost.keys())
    if missing_suppliers:
        raise ValueError(f'Missing transportation cost rows for suppliers: {missing_suppliers}')
    for s in suppliers:
        missing_cust = set(customers) - set(cost[s].keys())
        if missing_cust:
            raise ValueError(f'Missing transportation cost entries for supplier {s} and customers: {missing_cust}')
    missing_fixed = set(suppliers) - set(fixed_cost.keys())
    if missing_fixed:
        raise ValueError(f'Missing fixed cost entries for suppliers: {missing_fixed}')
    M = sum(demand.values())
    m = gp.Model('UFLP8')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()