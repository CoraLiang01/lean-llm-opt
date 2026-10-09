import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'demand': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', 'fixed_cost': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', 'transportation_costs': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'}

    def read_csv(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    demand_df = read_csv(csv_paths['demand'])
    fixed_cost_df = read_csv(csv_paths['fixed_cost'])
    trans_cost_df = read_csv(csv_paths['transportation_costs'])
    branches = list(demand_df['customer'])
    suppliers = list(fixed_cost_df['Unnamed: 0'])
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in demand_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing demand.csv: {e}')
    try:
        fixed_cost = {row['Unnamed: 0']: float(row['fixed_costs']) for (_, row) in fixed_cost_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing fixed_cost.csv: {e}')
    cost = {}
    branch_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    for (_, row) in trans_cost_df.iterrows():
        supplier = row['Unnamed: 0']
        cost[supplier] = {}
        for branch in branch_cols:
            if branch in branches:
                try:
                    cost[supplier][branch] = float(row[branch])
                except Exception as e:
                    raise ValueError(f'Error parsing transportation_costs.csv for supplier {supplier}, branch {branch}: {e}')
    for s in suppliers:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
        if s not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {s}')
        for b in branches:
            if b not in cost[s]:
                raise ValueError(f'Missing transportation cost for supplier {s}, branch {b}')
    for b in branches:
        if b not in demand:
            raise ValueError(f'Missing demand for branch {b}')
    m = gp.Model('Superstore_FLP')
    quantity_keys = [(s, b) for s in suppliers for b in branches]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[s][b] * quantity_vars[s, b] for s in suppliers for b in branches)) + gp.quicksum((fixed_cost[s] * activation_vars[s] for s in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[s, b] for s in suppliers)) == demand[b] for b in branches), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()