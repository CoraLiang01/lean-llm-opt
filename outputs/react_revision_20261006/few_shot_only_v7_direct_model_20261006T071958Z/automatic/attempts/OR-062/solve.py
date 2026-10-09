import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csvs = [{'table_id': 'file_0_view_0', 'path': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', 'columns': ['Customer', 'demand']}, {'table_id': 'file_1_view_0', 'path': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', 'columns': ['Unnamed: 0', 'fixed_costs']}, {'table_id': 'file_2_view_0', 'path': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'columns': ['Unnamed: 0', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']}]

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_df = read_csv_with_encodings(csvs[0]['path'], dtype=str, keep_default_na=False, usecols=csvs[0]['columns'])
    fixed_cost_df = read_csv_with_encodings(csvs[1]['path'], dtype=str, keep_default_na=False, usecols=csvs[1]['columns'])
    transportation_costs_df = read_csv_with_encodings(csvs[2]['path'], dtype=str, keep_default_na=False, usecols=csvs[2]['columns'])
    suppliers = fixed_cost_df['Unnamed: 0'].tolist()
    stores = demand_df['Customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        j = row['Customer']
        try:
            d = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for store {j}: {row['demand']}")
        if j in demand:
            demand[j] += d
        else:
            demand[j] = d
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        i = row['Unnamed: 0']
        try:
            f = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for supplier {i}: {row['fixed_costs']}")
        if i in fixed_cost:
            fixed_cost[i] += f
        else:
            fixed_cost[i] = f
    store_columns = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
    store_name_map = {}
    for col in store_columns:
        for j in stores:
            if col.casefold() == j.casefold():
                store_name_map[col] = j
                break
    for col in store_columns:
        if col not in store_name_map:
            raise ValueError(f"Store column '{col}' in transportation_costs.csv does not match any store in demand.csv.")
    cost = {}
    for (_, row) in transportation_costs_df.iterrows():
        i = row['Unnamed: 0']
        if i not in suppliers:
            continue
        cost[i] = {}
        for col in store_columns:
            j = store_name_map[col]
            try:
                cij = float(row[col])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {i}, store {col}: {row[col]}')
            cost[i][j] = cij
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, store {j}')
    m = gp.Model('Liquor_Distribution')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in suppliers for j in stores]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * activation_vars[i] for i in suppliers)), GRB.MINIMIZE)
    for j in stores:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()