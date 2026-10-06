import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path)
    fixed_cost_df = read_csv_with_encodings(fixed_cost_path)
    trans_cost_df = read_csv_with_encodings(trans_cost_path)
    if 'customer' in demand_df.columns:
        customers = demand_df['customer'].astype(str).tolist()
    else:
        raise ValueError("Missing 'customer' column in demand.csv")
    if 'Unnamed: 0' in fixed_cost_df.columns:
        suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
    else:
        raise ValueError("Missing 'Unnamed: 0' column in fixed_cost.csv")
    if 'demand' not in demand_df.columns:
        raise ValueError("Missing 'demand' column in demand.csv")
    demand = dict(zip(demand_df['customer'].astype(str), demand_df['demand']))
    if 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("Missing 'fixed_costs' column in fixed_cost.csv")
    fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str), fixed_cost_df['fixed_costs']))
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError("Missing 'Unnamed: 0' column in transportation_costs.csv")
    cost_customer_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Missing transportation cost columns for customers: {missing_customers}')
    missing_suppliers = set(suppliers) - set(trans_cost_df['Unnamed: 0'].astype(str))
    if missing_suppliers:
        raise ValueError(f'Missing transportation cost rows for suppliers: {missing_suppliers}')
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        supplier = str(row['Unnamed: 0'])
        cost[supplier] = {}
        for customer in customers:
            cost[supplier][customer] = row[customer]
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    M = sum((demand[j] for j in customers))
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()