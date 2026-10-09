import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cost_path} with tried encodings.')
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {demand_path} with tried encodings.')
    I = cost_df['plant'].tolist()
    J = demand_df['customer'].tolist()
    try:
        fixed_cost = {row['plant']: float(row['fixed_cost']) for (_, row) in cost_df.iterrows()}
    except Exception:
        raise ValueError('Error parsing fixed_cost in cost.csv')
    try:
        capacity = {row['plant']: float(row['capacity']) for (_, row) in cost_df.iterrows()}
    except Exception:
        raise ValueError('Error parsing capacity in cost.csv')
    c_ij = {}
    for (_, row) in cost_df.iterrows():
        i = row['plant']
        for j in J:
            if j not in cost_df.columns:
                raise ValueError(f'Customer {j} not found as column in cost.csv')
            try:
                c_ij[i, j] = float(row[j])
            except Exception:
                raise ValueError(f'Error parsing cost for plant {i}, customer {j} in cost.csv')
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in demand_df.iterrows()}
    except Exception:
        raise ValueError('Error parsing demand in demand.csv')
    if set(I) != set(cost_df['plant']):
        raise ValueError('Mismatch in plant indices between cost.csv and I')
    if set(J) != set(demand_df['customer']):
        raise ValueError('Mismatch in customer indices between demand.csv and J')
    for i in I:
        for j in J:
            if (i, j) not in c_ij:
                raise ValueError(f'Missing cost coefficient for plant {i}, customer {j}')
    m = gp.Model('UFLP')
    x_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in I for j in J)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= capacity[i] * y_vars[i] for i in I), name='')
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