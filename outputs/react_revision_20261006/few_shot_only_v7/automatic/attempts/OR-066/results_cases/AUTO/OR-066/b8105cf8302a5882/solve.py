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
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    fixed_cost_df = read_csv_robust(fixed_cost_path)
    trans_cost_df = read_csv_robust(transportation_costs_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        demand[cust] = val
    if 'Unnamed: 0' in fixed_cost_df.columns:
        supplier_col = 'Unnamed: 0'
    elif 'supplier' in fixed_cost_df.columns:
        supplier_col = 'supplier'
    else:
        raise ValueError('fixed_cost.csv must have a supplier index column')
    if 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have column 'fixed_costs'")
    suppliers = fixed_cost_df[supplier_col].tolist()
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = row[supplier_col]
        try:
            val = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for supplier {sup}: {row['fixed_costs']}")
        fixed_cost[sup] = val
    if 'Unnamed: 0' in trans_cost_df.columns:
        trans_supplier_col = 'Unnamed: 0'
    elif 'supplier' in trans_cost_df.columns:
        trans_supplier_col = 'supplier'
    else:
        raise ValueError('transportation_costs.csv must have a supplier index column')
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        sup = row[trans_supplier_col]
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Missing transportation cost for supplier {sup}, customer {cust}')
            try:
                val = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = val
    for sup in suppliers:
        if sup not in cost:
            raise ValueError(f'Supplier {sup} missing in transportation_costs.csv')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Customer {cust} missing for supplier {sup} in transportation_costs.csv')
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Customer {cust} missing in demand.csv')
    for sup in suppliers:
        if sup not in fixed_cost:
            raise ValueError(f'Supplier {sup} missing in fixed_cost.csv')
    x_keys = [(sup, cust) for sup in suppliers for cust in customers]
    y_keys = suppliers
    m = gp.Model('UFLP')
    quantity_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activate_vars = m.addVars(y_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[sup][cust] * quantity_vars[sup, cust] for (sup, cust) in x_keys)) + gp.quicksum((fixed_cost[sup] * activate_vars[sup] for sup in y_keys)), GRB.MINIMIZE)
    for cust in customers:
        m.addConstr(gp.quicksum((quantity_vars[sup, cust] for sup in suppliers)) == demand[cust], name=f'demand_{cust}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')