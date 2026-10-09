import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
    decode_attempts = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_attempts:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {capacity_path} with tried encodings.')
    for enc in decode_attempts:
        try:
            products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {products_path} with tried encodings.')
    I = capacity_df['StorageID'].tolist()
    J = products_df['ProductName'].tolist()
    try:
        c_i = capacity_df.set_index('StorageID')['Capacity'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    try:
        v_j = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Value to float: {e}')
    try:
        w_j = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Weight to float: {e}')
    missing_c = [i for i in I if i not in c_i]
    missing_v = [j for j in J if j not in v_j]
    missing_w = [j for j in J if j not in w_j]
    if missing_c:
        raise ValueError(f'Missing capacity for StorageID(s): {missing_c}')
    if missing_v:
        raise ValueError(f'Missing value for ProductName(s): {missing_v}')
    if missing_w:
        raise ValueError(f'Missing weight for ProductName(s): {missing_w}')
    m = gp.Model('Amazon_AC_Storage')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i], name=f'cap_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')