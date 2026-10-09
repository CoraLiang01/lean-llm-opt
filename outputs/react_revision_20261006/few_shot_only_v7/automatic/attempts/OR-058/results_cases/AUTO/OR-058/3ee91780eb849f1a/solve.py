import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
    try:
        demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding='latin-1')
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    customers = []
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        if cust not in customers:
            customers.append(cust)
        try:
            dval = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        if cust in demand:
            demand[cust] += dval
        else:
            demand[cust] = dval
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
    try:
        fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding='latin-1')
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    suppliers = []
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = row['Unnamed: 0']
        if sup not in suppliers:
            suppliers.append(sup)
        try:
            fval = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for supplier {sup}: {row['fixed_costs']}")
        if sup in fixed_cost:
            raise ValueError(f'Duplicate supplier in fixed_cost.csv: {sup}')
        fixed_cost[sup] = fval
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
    try:
        trans_cost_df = pd.read_csv(trans_cost_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            trans_cost_df = pd.read_csv(trans_cost_path, dtype=str, keep_default_na=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                trans_cost_df = pd.read_csv(trans_cost_path, dtype=str, keep_default_na=False, encoding='gbk')
            except UnicodeDecodeError:
                trans_cost_df = pd.read_csv(trans_cost_path, dtype=str, keep_default_na=False, encoding='latin-1')
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0'")
    cost_customer_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    for cust in customers:
        if cust not in cost_customer_cols:
            raise ValueError(f'Customer {cust} in demand.csv not found in transportation_costs.csv columns')
    for sup in suppliers:
        if sup not in list(trans_cost_df['Unnamed: 0']):
            raise ValueError(f'Supplier {sup} in fixed_cost.csv not found in transportation_costs.csv rows')
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        sup = row['Unnamed: 0']
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            try:
                cval = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = cval
    for sup in suppliers:
        if sup not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {sup}')
        if sup not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {sup}')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Missing transportation cost for supplier {sup}, customer {cust}')
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Missing demand for customer {cust}')
    m = gp.Model('UFLP')
    quantity_keys = [(sup, cust) for sup in suppliers for cust in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[sup][cust] * quantity_vars[sup, cust] for sup in suppliers for cust in customers)) + gp.quicksum((fixed_cost[sup] * open_vars[sup] for sup in suppliers)), GRB.MINIMIZE)
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