import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csvs = [('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', ['customer', 'demand']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', ['Unnamed: 0', 'fixed_costs']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', None)]

    def read_csv_with_encodings(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    demand_df = read_csv_with_encodings(csvs[0][0])
    if set(csvs[0][1]) - set(demand_df.columns):
        raise ValueError(f'Missing columns in demand.csv: {set(csvs[0][1]) - set(demand_df.columns)}')
    demand_df['demand'] = pd.to_numeric(demand_df['demand'], errors='raise')
    customers = demand_df['customer'].tolist()
    demand = dict(zip(demand_df['customer'], demand_df['demand']))
    fixed_cost_df = read_csv_with_encodings(csvs[1][0])
    if set(csvs[1][1]) - set(fixed_cost_df.columns):
        raise ValueError(f'Missing columns in fixed_cost.csv: {set(csvs[1][1]) - set(fixed_cost_df.columns)}')
    fixed_cost_df['fixed_costs'] = pd.to_numeric(fixed_cost_df['fixed_costs'], errors='raise')
    warehouses = fixed_cost_df['Unnamed: 0'].tolist()
    fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
    trans_cost_df = read_csv_with_encodings(csvs[2][0])
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError("Missing 'Unnamed: 0' in transportation_costs.csv")
    missing_customers = set(customers) - set(trans_cost_df.columns)
    if missing_customers:
        raise ValueError(f'Missing customer columns in transportation_costs.csv: {missing_customers}')
    missing_warehouses = set(warehouses) - set(trans_cost_df['Unnamed: 0'])
    if missing_warehouses:
        raise ValueError(f'Missing warehouse rows in transportation_costs.csv: {missing_warehouses}')
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        w = row['Unnamed: 0']
        if w not in warehouses:
            continue
        cost[w] = {}
        for c in customers:
            try:
                cost[w][c] = float(row[c])
            except Exception:
                raise ValueError(f'Invalid cost for warehouse {w}, customer {c} in transportation_costs.csv')
    for w in warehouses:
        if w not in fixed_cost:
            raise ValueError(f'Warehouse {w} missing in fixed_cost.csv')
        if w not in cost:
            raise ValueError(f'Warehouse {w} missing in transportation_costs.csv')
        for c in customers:
            if c not in cost[w]:
                raise ValueError(f'Cost missing for warehouse {w}, customer {c} in transportation_costs.csv')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Customer {c} missing in demand.csv')
    m = gp.Model('Bandcamp_FLP')
    quantity_keys = [(w, c) for w in warehouses for c in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w][c] * quantity_vars[w, c] for (w, c) in quantity_keys)) + gp.quicksum((fixed_cost[w] * activation_vars[w] for w in warehouses)), GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((quantity_vars[w, c] for w in warehouses)) == demand[c], name=f'demand_{c}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')