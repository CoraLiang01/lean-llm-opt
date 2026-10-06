import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_try_encodings(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
    demand_df = read_csv_try_encodings(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].astype(str).tolist()
    demand = demand_df.groupby('customer')['demand'].sum().to_dict()
    fixed_cost_df = read_csv_try_encodings(fixed_cost_path)
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    warehouses = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
    fixed_cost = fixed_cost_df.set_index('Unnamed: 0')['fixed_costs'].to_dict()
    trans_cost_df = read_csv_try_encodings(transportation_costs_path)
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0'")
    cost_customer_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Missing transportation cost columns for customers: {missing_customers}')
    missing_warehouses = set(warehouses) - set(trans_cost_df['Unnamed: 0'].astype(str))
    if missing_warehouses:
        raise ValueError(f'Missing transportation cost rows for warehouses: {missing_warehouses}')
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        w = str(row['Unnamed: 0'])
        cost[w] = {}
        for c in customers:
            cost[w][c] = row[c]
    for w in warehouses:
        if w not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {w}')
        if w not in cost:
            raise ValueError(f'Missing cost row for warehouse {w}')
        for c in customers:
            if c not in cost[w]:
                raise ValueError(f'Missing transportation cost for warehouse {w}, customer {c}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    M = sum(demand.values())
    m = gp.Model('Bandcamp_FLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w][c] * x[w, c] for w in warehouses for c in customers)) + gp.quicksum((fixed_cost[w] * y[w] for w in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[w, c] for w in warehouses)) == demand[c] for c in customers), name='')
    m.addConstrs((gp.quicksum((x[w, c] for c in customers)) <= M * y[w] for w in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()