import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
    demand_df = None
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    if demand_df is None:
        raise RuntimeError(f'Failed to read {demand_path} with supported encodings.')
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError('demand.csv missing required columns.')
    customers = demand_df['customer'].tolist()
    demand_dict = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            demand_val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        if cust in demand_dict:
            demand_dict[cust] += demand_val
        else:
            demand_dict[cust] = demand_val
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
    fixed_cost_df = None
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    if fixed_cost_df is None:
        raise RuntimeError(f'Failed to read {fixed_cost_path} with supported encodings.')
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError('fixed_cost.csv missing required columns.')
    warehouses = fixed_cost_df['Unnamed: 0'].tolist()
    fixed_cost_dict = {}
    for (_, row) in fixed_cost_df.iterrows():
        wh = row['Unnamed: 0']
        try:
            fc = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed_cost for warehouse {wh}: {row['fixed_costs']}")
        if wh in fixed_cost_dict:
            fixed_cost_dict[wh] += fc
        else:
            fixed_cost_dict[wh] = fc
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
    trans_cost_df = None
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            trans_cost_df = pd.read_csv(trans_cost_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    if trans_cost_df is None:
        raise RuntimeError(f'Failed to read {trans_cost_path} with supported encodings.')
    if 'Unnamed: 0' not in trans_cost_df.columns:
        raise ValueError('transportation_costs.csv missing required row identifier column.')
    cost_dict = {}
    for (_, row) in trans_cost_df.iterrows():
        wh = row['Unnamed: 0']
        if wh not in warehouses:
            continue
        cost_dict[wh] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Customer {cust} missing in transportation_costs.csv for warehouse {wh}.')
            try:
                cij = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for warehouse {wh}, customer {cust}: {row[cust]}')
            cost_dict[wh][cust] = cij
    for wh in warehouses:
        if wh not in fixed_cost_dict:
            raise ValueError(f'Warehouse {wh} missing fixed cost.')
        if wh not in cost_dict:
            raise ValueError(f'Warehouse {wh} missing in transportation_costs.csv.')
        for cust in customers:
            if cust not in cost_dict[wh]:
                raise ValueError(f'Transportation cost missing for warehouse {wh}, customer {cust}.')
    for cust in customers:
        if cust not in demand_dict:
            raise ValueError(f'Customer {cust} missing demand.')
    total_demand = sum((demand_dict[cust] for cust in customers))
    M_dict = {wh: total_demand for wh in warehouses}
    m = gp.Model('Bandcamp_FLP')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(wh, cust) for wh in warehouses for cust in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost_dict[wh][cust] * quantity_vars[wh, cust] for wh in warehouses for cust in customers)) + gp.quicksum((fixed_cost_dict[wh] * activation_vars[wh] for wh in warehouses)), GRB.MINIMIZE)
    for cust in customers:
        m.addConstr(gp.quicksum((quantity_vars[wh, cust] for wh in warehouses)) == demand_dict[cust], name='demand_' + str(cust))
    for wh in warehouses:
        m.addConstr(gp.quicksum((quantity_vars[wh, cust] for cust in customers)) <= M_dict[wh] * activation_vars[wh], name='activation_' + str(wh))
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')