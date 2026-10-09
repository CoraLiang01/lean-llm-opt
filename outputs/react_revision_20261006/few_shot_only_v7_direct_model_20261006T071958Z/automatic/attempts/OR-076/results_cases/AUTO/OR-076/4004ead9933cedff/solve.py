import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
    warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {cost_path} with tried encodings.')
    for enc in encodings:
        try:
            warehouse_df = pd.read_csv(warehouse_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {warehouse_path} with tried encodings.')
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {demand_path} with tried encodings.')
    I = warehouse_df['Warehouse ID'].tolist()
    J = demand_df['Customer ID'].tolist()
    try:
        f_i = {}
        for (_, row) in warehouse_df.iterrows():
            i = row['Warehouse ID']
            try:
                f_i[i] = float(row['Fixed_Cost'])
            except Exception:
                raise ValueError(f"Invalid Fixed_Cost for warehouse {i}: {row['Fixed_Cost']}")
    except Exception as e:
        raise RuntimeError(f'Error processing Fixed_Cost: {e}')
    try:
        cap_i = {}
        for (_, row) in warehouse_df.iterrows():
            i = row['Warehouse ID']
            try:
                cap_i[i] = float(row['Capacity'])
            except Exception:
                raise ValueError(f"Invalid Capacity for warehouse {i}: {row['Capacity']}")
    except Exception as e:
        raise RuntimeError(f'Error processing Capacity: {e}')
    try:
        d_j = {}
        for (_, row) in demand_df.iterrows():
            j = row['Customer ID']
            try:
                d_j[j] = float(row['Demand'])
            except Exception:
                raise ValueError(f"Invalid Demand for customer {j}: {row['Demand']}")
    except Exception as e:
        raise RuntimeError(f'Error processing Demand: {e}')
    cost_columns = [col for col in cost_df.columns if col != 'Warehouse ID']
    missing_customers = [j for j in J if j not in cost_columns]
    if missing_customers:
        raise ValueError(f'Customer IDs {missing_customers} not found in cost.csv columns.')
    c_ij = {}
    for (_, row) in cost_df.iterrows():
        i = row['Warehouse ID']
        if i not in I:
            continue
        c_ij[i] = {}
        for j in J:
            try:
                c_ij[i][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid cost for warehouse {i}, customer {j}: {row[j]}')
    for i in I:
        if i not in c_ij:
            raise ValueError(f'Warehouse {i} missing in cost.csv.')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Cost for warehouse {i}, customer {j} missing in cost.csv.')
    m = gp.Model('UFLP4')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * quantity_vars[i, j] for (i, j) in quantity_keys)) + gp.quicksum((f_i[i] * open_vars[i] for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in I)) == d_j[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for j in J)) <= cap_i[i] * open_vars[i], name=f'capacity_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()