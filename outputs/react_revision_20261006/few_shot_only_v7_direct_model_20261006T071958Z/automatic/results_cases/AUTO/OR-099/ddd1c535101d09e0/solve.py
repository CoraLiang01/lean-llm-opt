import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
    store_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_multi_enc(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    warehouse_df = read_csv_multi_enc(warehouse_path)
    store_df = read_csv_multi_enc(store_path)
    cost_df = read_csv_multi_enc(cost_path)
    I = warehouse_df['Warehouse (i)'].tolist()
    J = store_df['Store (j)'].tolist()
    f_i = {}
    for (idx, row) in warehouse_df.iterrows():
        wid = row['Warehouse (i)']
        try:
            f_i[wid] = float(row['Opening Cost (fi)'])
        except Exception:
            raise ValueError(f"Invalid opening cost for warehouse {wid}: {row['Opening Cost (fi)']}")
    u_i = {}
    for (idx, row) in warehouse_df.iterrows():
        wid = row['Warehouse (i)']
        try:
            u_i[wid] = float(row['Capacity (units)'])
        except Exception:
            raise ValueError(f"Invalid capacity for warehouse {wid}: {row['Capacity (units)']}")
    d_j = {}
    for (idx, row) in store_df.iterrows():
        sid = row['Store (j)']
        try:
            d_j[sid] = float(row['Demand (units, dj)'])
        except Exception:
            raise ValueError(f"Invalid demand for store {sid}: {row['Demand (units, dj)']}")
    c_ij = {}
    cost_row_ids = cost_df['Unnamed: 0'].tolist()
    cost_col_ids = [col for col in cost_df.columns if col != 'Unnamed: 0']
    missing_warehouses = [i for i in I if i not in cost_row_ids]
    missing_stores = [j for j in J if j not in cost_col_ids]
    if missing_warehouses:
        raise ValueError(f'Missing warehouses in TransportationCost.csv: {missing_warehouses}')
    if missing_stores:
        raise ValueError(f'Missing stores in TransportationCost.csv: {missing_stores}')
    for i in I:
        c_ij[i] = {}
        row = cost_df[cost_df['Unnamed: 0'] == i]
        if row.empty:
            raise ValueError(f'Warehouse {i} not found in TransportationCost.csv')
        row = row.iloc[0]
        for j in J:
            try:
                c_ij[i][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid transportation cost for warehouse {i}, store {j}: {row[j]}')
    if set(f_i.keys()) != set(I):
        raise ValueError('Mismatch in warehouse opening cost keys and I')
    if set(u_i.keys()) != set(I):
        raise ValueError('Mismatch in warehouse capacity keys and I')
    if set(d_j.keys()) != set(J):
        raise ValueError('Mismatch in store demand keys and J')
    for i in I:
        if set(c_ij[i].keys()) != set(J):
            raise ValueError(f'Mismatch in transportation cost keys for warehouse {i} and J')
    m = gp.Model('UFLP14')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f_i[i] * open_vars[i] for i in I)) + gp.quicksum((c_ij[i][j] * quantity_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in J)) <= u_i[i] * open_vars[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()