import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
    customer_demand_df = read_csv_with_encodings(customer_demand_path)
    supply_capacity_df = read_csv_with_encodings(supply_capacity_path)
    transportation_costs_df = read_csv_with_encodings(transportation_costs_path)
    I = supply_capacity_df['Unnamed: 0'].tolist()
    J = customer_demand_df['customer'].tolist()
    try:
        d_j = {row['customer']: float(row['demand']) for (_, row) in customer_demand_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing demand values: {e}')
    try:
        s_i = {row['Unnamed: 0']: float(row['supply_capacity']) for (_, row) in supply_capacity_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing supply_capacity values: {e}')
    cost_df = transportation_costs_df.set_index('Unnamed: 0')
    missing_costs = []
    c_ij = {}
    for i in I:
        if i not in cost_df.index:
            missing_costs.append((i, 'ALL'))
            continue
        c_ij[i] = {}
        for j in J:
            if j not in cost_df.columns:
                missing_costs.append((i, j))
                continue
            val = cost_df.at[i, j]
            try:
                c_ij[i][j] = float(val)
            except Exception:
                missing_costs.append((i, j))
    if missing_costs:
        raise ValueError(f'Missing or invalid transportation cost(s) for: {missing_costs}')
    m = gp.Model('TP4_Original_RAG')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * quantity_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) >= d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in J)) <= s_i[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()