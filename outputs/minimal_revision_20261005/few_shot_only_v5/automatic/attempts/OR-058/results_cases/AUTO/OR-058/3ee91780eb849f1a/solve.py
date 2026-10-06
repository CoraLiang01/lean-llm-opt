import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
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
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].astype(str).tolist()
    demand = demand_df.groupby('customer')['demand'].sum().to_dict()
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
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
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
    fixed_cost = fixed_cost_df.set_index('Unnamed: 0')['fixed_costs'].to_dict()
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
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
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
    missing_customers = [c for c in customers if c not in trans_cost_df.columns]
    if missing_customers:
        raise ValueError(f'transportation_costs.csv missing columns for customers: {missing_customers}')
    missing_suppliers = [s for s in suppliers if s not in trans_cost_df['Unnamed: 0'].astype(str).tolist()]
    if missing_suppliers:
        raise ValueError(f'transportation_costs.csv missing rows for suppliers: {missing_suppliers}')
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        supplier = str(row['Unnamed: 0'])
        if supplier not in suppliers:
            continue
        cost[supplier] = {}
        for customer in customers:
            cost[supplier][customer] = float(row[customer])
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    M = sum((demand[j] for j in customers))
    m = gp.Model('UFLP_Adidas')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in suppliers for j in customers]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')