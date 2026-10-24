import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    J = [1, 2, 3]
    I = list(range(1, 102))
    t = {j: {} for j in J}
    m_percent = {}
    hi_fi_cols = [f'HiFi{i}_Minutes' for i in I]
    missing_cols = [col for col in hi_fi_cols + ['Workstation', 'Maintenance_Percent'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns in CSV: {missing_cols}')
    for (_, row) in df.iterrows():
        ws_str = row['Workstation'].strip()
        if not ws_str.isdigit():
            continue
        j = int(ws_str)
        if j not in J:
            continue
        m_val_str = row['Maintenance_Percent'].strip()
        if m_val_str == '':
            raise ValueError(f'Empty Maintenance_Percent for workstation {j}')
        try:
            m_val = float(m_val_str)
        except ValueError:
            raise ValueError(f'Invalid Maintenance_Percent "{m_val_str}" for workstation {j}')
        m_percent[j] = m_val
        for i in I:
            col = f'HiFi{i}_Minutes'
            val_str = row[col].strip()
            if val_str == '':
                raise ValueError(f'Empty time for HiFi{i} on workstation {j}')
            try:
                val = int(val_str)
            except ValueError:
                raise ValueError(f'Invalid time "{val_str}" for HiFi{i} on workstation {j}')
            t[j][i] = val
    for j in J:
        if j not in m_percent:
            raise ValueError(f'Missing maintenance percent for workstation {j}')
        if len(t[j]) != len(I):
            missing_i = [i for i in I if i not in t[j]]
            raise ValueError(f'Missing times for HiFi models {missing_i} on workstation {j}')
    C = {j: 1440 * (1 - m_percent[j] / 100) for j in J}
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    for j in J:
        expr = gurobipy.quicksum((t[j][i] * x[i] for i in I))
        m.addConstr(expr <= C[j], name=f'cap_{j}')
    const_term = gurobipy.quicksum((C[j] for j in J))
    prod_term = gurobipy.quicksum((t[j][i] * x[i] for j in J for i in I))
    m.setObjective(const_term - prod_term, GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()