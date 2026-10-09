import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path, dtype=str, keep_default_na=False)
    fixed_cost_df = read_csv_with_encodings(fixed_cost_path, dtype=str, keep_default_na=False)
    trans_cost_df = read_csv_with_encodings(trans_cost_path, dtype=str, keep_default_na=False)
    suppliers = pd.unique(pd.concat([fixed_cost_df['Unnamed: 0'], trans_cost_df['Unnamed: 0']], ignore_index=True)).tolist()
    stores = pd.unique(pd.concat([demand_df['customer'], pd.Series([col for col in trans_cost_df.columns if col.startswith('C')])], ignore_index=True)).tolist()
    demand_dict = {}
    for (_, row) in demand_df.iterrows():
        store = row['customer']
        if store in demand_dict:
            raise ValueError(f'Duplicate demand entry for store {store}')
        try:
            demand_dict[store] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for store {store}: {row['demand']}")
    fixed_cost_dict = {}
    for (_, row) in fixed_cost_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier in fixed_cost_dict:
            raise ValueError(f'Duplicate fixed cost entry for supplier {supplier}')
        try:
            fixed_cost_dict[supplier] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for supplier {supplier}: {row['fixed_costs']}")
    cost_dict = {supplier: {} for supplier in suppliers}
    for (_, row) in trans_cost_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            continue
        for store in stores:
            if store in trans_cost_df.columns:
                val = row[store]
            else:
                val = row.get(store, None)
            if store in trans_cost_df.columns:
                try:
                    cost_dict[supplier][store] = float(row[store])
                except Exception:
                    raise ValueError(f'Invalid transportation cost for ({supplier},{store}): {row[store]}')
            elif store in [col for col in trans_cost_df.columns if col.startswith('C')]:
                try:
                    cost_dict[supplier][store] = float(row[store])
                except Exception:
                    continue
            else:
                continue
    for i in suppliers:
        for j in stores:
            if j not in cost_dict[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, store {j}')
    for i in suppliers:
        if i not in fixed_cost_dict:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in stores:
        if j not in demand_dict:
            raise ValueError(f'Missing demand for store {j}')
    M = sum((demand_dict[j] for j in stores))
    m = gp.Model('UFLP_Adidas')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost_dict[i][j] * quantity_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost_dict[i] * open_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand_dict[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in stores)) <= M * open_vars[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()