import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', dtype=str, keep_default_na=False)
demand = {}
customers = []
for idx, row in demand_df.iterrows():
    cust = row['Customer'].strip() if 'Customer' in row else row[0].strip()
    val = row['demand'] if 'demand' in row else row[1]
    cust_name = cust
    try:
        demand_val = float(val)
    except Exception:
        raise ValueError(f'Invalid demand value for {cust}: {val}')
    demand[cust_name] = demand_val
    customers.append(cust_name)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', dtype=str, keep_default_na=False)
fixed_cost = {}
suppliers = []
for idx, row in fixed_cost_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    val = row['fixed_costs']
    try:
        cost_val = float(val)
    except Exception:
        raise ValueError(f'Invalid fixed cost for {sup}: {val}')
    fixed_cost[sup] = cost_val
    suppliers.append(sup)
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', dtype=str, keep_default_na=False)
cost = {}
customer_cols = [col.strip() for col in trans_costs_df.columns if col.strip() != 'Unnamed: 0']
for idx, row in trans_costs_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    cost[sup] = {}
    for cust in customer_cols:
        val = row[cust]
        try:
            cost_val = float(val)
        except Exception:
            raise ValueError(f'Invalid transportation cost for {sup} to {cust}: {val}')
        cost[sup][cust] = cost_val
customer_map = {'Customer_1': 'CLARINDA', 'Customer_2': 'FORT MADISON', 'Customer_3': 'SIOUX CITY', 'Customer_4': 'TOLEDO', 'Customer_5': 'BANCROFT'}
customer_map_rev = {v: k for k, v in customer_map.items()}
for cust in customer_cols:
    if cust not in customer_map_rev:
        raise ValueError(f'Customer {cust} in transportation_costs.csv not found in demand.csv mapping.')
customers = [customer_map_rev[cust] for cust in customer_cols]
demand_costfile = {}
for cust in customer_cols:
    demand_costfile[cust] = demand[customer_map_rev[cust]]
M = sum(demand_costfile.values())
for sup in suppliers:
    if sup not in fixed_cost:
        raise ValueError(f'Supplier {sup} missing from fixed_cost.')
    if sup not in cost:
        raise ValueError(f'Supplier {sup} missing from transportation_costs.')
    for cust in customer_cols:
        if cust not in cost[sup]:
            raise ValueError(f'Cost missing for {sup} to {cust}.')
m = gp.Model('Iowa_Liquor_UFLP')
x_vars = m.addVars(suppliers, customer_cols, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[sup][cust] * x_vars[sup, cust] for sup in suppliers for cust in customer_cols)) + gp.quicksum((fixed_cost[sup] * y_vars[sup] for sup in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand_costfile[cust] for cust in customer_cols), name='')
m.addConstrs((gp.quicksum((x_vars[sup, cust] for cust in customer_cols)) <= M * y_vars[sup] for sup in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')