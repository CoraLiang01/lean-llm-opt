import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
    demand_df = read_csv(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    J = demand_df['customer'].tolist()
    d_j = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            demand_val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        d_j[cust] = demand_val
    fixed_cost_df = read_csv(fixed_cost_path)
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    I = fixed_cost_df['Unnamed: 0'].tolist()
    f_i = {}
    for (_, row) in fixed_cost_df.iterrows():
        fac = row['Unnamed: 0']
        try:
            fixed_val = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed_costs for supplier {fac}: {row['fixed_costs']}")
        f_i[fac] = fixed_val
    trans_df = read_csv(transportation_costs_path)
    if 'Unnamed: 0' not in trans_df.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0'")
    cost_cols = [col for col in trans_df.columns if col != 'Unnamed: 0']
    if len(cost_cols) != len(J):
        raise ValueError('Number of columns in transportation_costs.csv does not match number of customers in demand.csv')
    c_ij = {}
    for (_, row) in trans_df.iterrows():
        i = row['Unnamed: 0']
        if i not in I:
            continue
        c_ij[i] = {}
        for (idx, j) in enumerate(J):
            col = cost_cols[idx]
            try:
                cij_val = float(row[col])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {i}, customer {j}: {row[col]}')
            c_ij[i][j] = cij_val
    for i in I:
        if i not in c_ij:
            raise ValueError(f'Supplier {i} missing in transportation_costs.csv')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Cost for supplier {i}, customer {j} missing in transportation_costs.csv')
    M = sum((d_j[j] for j in J))
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in I for j in J]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * x_vars[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * y_vars[i] for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == d_j[j], name='demand_' + str(j))
    for i in I:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in J)) <= M * y_vars[i], name='open_' + str(i))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()