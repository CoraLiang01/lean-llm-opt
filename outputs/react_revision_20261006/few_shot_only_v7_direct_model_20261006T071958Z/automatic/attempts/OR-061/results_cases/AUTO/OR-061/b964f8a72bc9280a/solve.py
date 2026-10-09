import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_enc(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
    demand_df = read_csv_enc(demand_path)
    fixed_cost_df = read_csv_enc(fixed_cost_path)
    trans_cost_df = read_csv_enc(transportation_costs_path)
    suppliers = fixed_cost_df['Unnamed: 0'].tolist()
    branches = demand_df['customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        branch = row['customer']
        try:
            demand[branch] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for branch {branch}: {row['demand']}")
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        supplier = row['Unnamed: 0']
        try:
            fixed_cost[supplier] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed_cost value for supplier {supplier}: {row['fixed_costs']}")
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        supplier = row['Unnamed: 0']
        cost[supplier] = {}
        for branch in branches:
            if branch not in row:
                raise ValueError(f'Branch {branch} not found in transportation_costs.csv columns.')
            try:
                cost[supplier][branch] = float(row[branch])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {supplier}, branch {branch}: {row[branch]}')
    D = dict(demand)
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in branches:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, branch {j}')
    for j in branches:
        if j not in demand:
            raise ValueError(f'Missing demand for branch {j}')
    m = gp.Model('Superstore_FLP')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in suppliers for j in branches]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for (i, j) in quantity_keys)) + gp.quicksum((fixed_cost[i] * activation_vars[i] for i in suppliers)), GRB.MINIMIZE)
    for j in branches:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
    for i in suppliers:
        for j in branches:
            m.addConstr(quantity_vars[i, j] <= D[j] * activation_vars[i], name=f'activation_{i}_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()