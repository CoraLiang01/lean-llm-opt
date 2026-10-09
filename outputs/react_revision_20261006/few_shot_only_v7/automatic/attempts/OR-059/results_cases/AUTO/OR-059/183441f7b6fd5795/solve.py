import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path, dtype=str, keep_default_na=False)
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
    fixed_cost_df = read_csv_robust(fixed_cost_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    suppliers = fixed_cost_df['Unnamed: 0'].tolist()
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = row['Unnamed: 0']
        try:
            val = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for supplier {sup}: {row['fixed_costs']}")
        fixed_cost[sup] = val
    trans_cost_df = read_csv_robust(transportation_costs_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for supplier IDs")
    cost_customer_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Customers {missing_customers} in demand.csv not found in transportation_costs.csv columns')
    missing_suppliers = set(suppliers) - set(trans_cost_df['Unnamed: 0'])
    if missing_suppliers:
        raise ValueError(f'Suppliers {missing_suppliers} in fixed_cost.csv not found in transportation_costs.csv rows')
    cost = {sup: {} for sup in suppliers}
    for (_, row) in trans_cost_df.iterrows():
        sup = row['Unnamed: 0']
        if sup not in suppliers:
            continue
        for cust in customers:
            try:
                val = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = val
    M = sum((demand[cust] for cust in customers))
    M_i = {sup: M for sup in suppliers}
    for sup in suppliers:
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Missing transportation cost for supplier {sup}, customer {cust}')
    m = gp.Model('ColoradoMotorVehicleSales_UFLP')
    m.Params.MIPGap = 0.0001
    x_keys = [(sup, cust) for sup in suppliers for cust in customers]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[sup][cust] * x_vars[sup, cust] for sup in suppliers for cust in customers)) + gp.quicksum((fixed_cost[sup] * y_vars[sup] for sup in suppliers)), GRB.MINIMIZE)
    for cust in customers:
        m.addConstr(gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand[cust], name=f'demand_{cust}')
    for sup in suppliers:
        m.addConstr(gp.quicksum((x_vars[sup, cust] for cust in customers)) <= M_i[sup] * y_vars[sup], name=f'activation_{sup}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()