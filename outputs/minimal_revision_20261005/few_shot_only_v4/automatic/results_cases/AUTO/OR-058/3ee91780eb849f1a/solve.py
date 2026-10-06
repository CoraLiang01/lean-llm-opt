import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    fixed_cost_df = read_csv_robust(fixed_cost_path)
    trans_cost_df = read_csv_robust(trans_cost_path)
    if 'Unnamed: 0' in fixed_cost_df.columns:
        suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
    else:
        raise ValueError("fixed_cost.csv missing supplier ID column 'Unnamed: 0'")
    if 'customer' in demand_df.columns:
        stores = demand_df['customer'].astype(str).tolist()
    else:
        raise ValueError("demand.csv missing store ID column 'customer'")
    if not set(stores).issubset(set(demand_df['customer'].astype(str))):
        raise ValueError('Mismatch between store IDs in demand.csv and expected stores.')
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = str(row['customer'])
        if key in demand:
            demand[key] += float(row['demand'])
        else:
            demand[key] = float(row['demand'])
    if not set(suppliers).issubset(set(fixed_cost_df['Unnamed: 0'].astype(str))):
        raise ValueError('Mismatch between supplier IDs in fixed_cost.csv and expected suppliers.')
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        key = str(row['Unnamed: 0'])
        fixed_cost[key] = float(row['fixed_costs'])
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError("transportation_costs.csv missing supplier ID column 'Unnamed: 0'")
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        sup = str(row['Unnamed: 0'])
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for store in stores:
            if store not in row:
                found = False
                for col in row.index:
                    if col.casefold() == store.casefold():
                        cost[sup][store] = float(row[col])
                        found = True
                        break
                if not found:
                    raise ValueError(f'Store {store} not found in transportation_costs.csv columns.')
            else:
                cost[sup][store] = float(row[store])
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Supplier {i} missing in transportation_costs.csv.')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Cost for supplier {i}, store {j} missing in transportation_costs.csv.')
    M = sum((demand[j] for j in stores))
    m = gp.Model('UFLP_Adidas')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()