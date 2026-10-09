import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except Exception:
                continue
        raise RuntimeError(f'Could not read {path} with supported encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            dval = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        demand[cust] = dval
    fixed_cost_df = read_csv_robust(fixed_cost_path)
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    warehouses = fixed_cost_df['Unnamed: 0'].tolist()
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        fac = row['Unnamed: 0']
        try:
            fval = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for warehouse {fac}: {row['fixed_costs']}")
        fixed_cost[fac] = fval
    trans_costs_df = read_csv_robust(transportation_costs_path)
    if 'Unnamed: 0' not in trans_costs_df.columns:
        raise ValueError("transportation_costs.csv must have 'Unnamed: 0' as warehouse/facility index")
    cost_customers = [col for col in trans_costs_df.columns if col != 'Unnamed: 0']
    if set(cost_customers) != set(customers):
        raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers)} vs {set(cost_customers)}')
    if set(trans_costs_df['Unnamed: 0']) != set(warehouses):
        raise ValueError(f"Warehouse mismatch between fixed_cost.csv and transportation_costs.csv: {set(warehouses)} vs {set(trans_costs_df['Unnamed: 0'])}")
    cost = {}
    for (_, row) in trans_costs_df.iterrows():
        fac = row['Unnamed: 0']
        cost[fac] = {}
        for cust in customers:
            try:
                cval = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for warehouse {fac}, customer {cust}: {row[cust]}')
            cost[fac][cust] = cval
    I = warehouses
    J = customers
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for warehouse {i}')
        for j in J:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i}, customer {j}')
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('Bandcamp_UFLP')
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in I for j in J)) + gp.quicksum((fixed_cost[i] * activation_vars[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) == demand[j] for j in J), name='')
    m.addConstrs((quantity_vars[i, j] <= demand[j] * activation_vars[i] for i in I for j in J), name='')
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