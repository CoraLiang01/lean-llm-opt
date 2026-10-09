import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
    try:
        demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='latin-1')
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
    try:
        fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='latin-1')
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
    try:
        transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False, encoding='latin-1')
    I = list(fixed_cost_df['Unnamed: 0'])
    J = list(demand_df['customer'])
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in demand_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing demand values: {e}')
    try:
        fixed_cost = {row['Unnamed: 0']: float(row['fixed_costs']) for (_, row) in fixed_cost_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing fixed cost values: {e}')
    cost = {}
    for (_, row) in transportation_costs_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in I:
            continue
        cost[supplier] = {}
        for j in J:
            if j not in transportation_costs_df.columns:
                raise ValueError(f'Supermarket {j} not found in transportation_costs.csv columns')
            try:
                cost[supplier][j] = float(row[j])
            except Exception as e:
                raise ValueError(f'Error parsing transportation cost for supplier {supplier}, supermarket {j}: {e}')
    M = sum((demand[j] for j in J))
    M_i = {i: M for i in I}
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in J:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, supermarket {j}')
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for supermarket {j}')
    m = gp.Model('UFLP8')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in I for j in J)) + gp.quicksum((fixed_cost[i] * open_vars[i] for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in I)) == demand[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for j in J)) <= M_i[i] * open_vars[i], name=f'activation_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()