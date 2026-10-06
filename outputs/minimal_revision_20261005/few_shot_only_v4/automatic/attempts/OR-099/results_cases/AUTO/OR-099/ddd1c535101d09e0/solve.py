import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    wh_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
    st_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
    tc_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
    wh_df = read_csv_with_encodings(wh_path)
    st_df = read_csv_with_encodings(st_path)
    tc_df = read_csv_with_encodings(tc_path)
    wh_col = [col for col in wh_df.columns if 'Warehouse' in col][0]
    fi_col = [col for col in wh_df.columns if 'Opening Cost' in col][0]
    ki_col = [col for col in wh_df.columns if 'Capacity' in col][0]
    warehouses = list(wh_df[wh_col])
    f = dict(zip(wh_df[wh_col], wh_df[fi_col]))
    K = dict(zip(wh_df[wh_col], wh_df[ki_col]))
    st_col = [col for col in st_df.columns if 'Store' in col][0]
    dj_col = [col for col in st_df.columns if 'Demand' in col][0]
    stores = list(st_df[st_col])
    d = dict(zip(st_df[st_col], st_df[dj_col]))
    tc_df = tc_df.rename(columns={tc_df.columns[0]: 'Warehouse'})
    tc_df.set_index('Warehouse', inplace=True)
    wh_row_map = dict(zip(tc_df.index, warehouses))
    st_col_map = dict(zip([str(i + 1) for i in range(len(stores))], stores))
    c = {}
    for wh_row in tc_df.index:
        i = wh_row_map[wh_row]
        for st_col in tc_df.columns:
            if st_col not in st_col_map:
                continue
            j = st_col_map[st_col]
            c[i, j] = tc_df.loc[wh_row, st_col]
    for i in warehouses:
        for j in stores:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
    for i in warehouses:
        if i not in f or i not in K:
            raise ValueError(f'Missing opening cost or capacity for warehouse {i}')
    for j in stores:
        if j not in d:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('UFLP14')
    x = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in warehouses)) + gp.quicksum((c[i, j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == d[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= K[i] * y[i] for i in warehouses), name='')
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