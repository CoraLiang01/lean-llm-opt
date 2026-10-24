import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
    cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
    demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    cost_df['fixed_cost'] = cost_df['fixed_cost'].astype(float)
    cost_df['capacity'] = cost_df['capacity'].astype(float)
    customer_cols = [col for col in cost_df.columns if col.startswith('C')]
    for col in customer_cols:
        cost_df[col] = cost_df[col].astype(float)
    demand_df['demand'] = demand_df['demand'].astype(float)
    plants = cost_df['plant'].tolist()
    customers = customer_cols
    demand_customers = demand_df['customer'].tolist()
    if set(customers) != set(demand_customers):
        missing_in_cost = set(demand_customers) - set(customers)
        missing_in_demand = set(customers) - set(demand_customers)
        msg = ''
        if missing_in_cost:
            msg += f'Customers {missing_in_cost} missing in cost data. '
        if missing_in_demand:
            msg += f'Customers {missing_in_demand} missing in demand data.'
        raise ValueError(msg)
    fixed_cost = dict(zip(cost_df['plant'], cost_df['fixed_cost']))
    capacity = dict(zip(cost_df['plant'], cost_df['capacity']))
    transport_cost = {}
    for (_, row) in cost_df.iterrows():
        p = row['plant']
        transport_cost[p] = {c: row[c] for c in customers}
    demand = dict(zip(demand_df['customer'], demand_df['demand']))
    for p in plants:
        if p not in fixed_cost or p not in capacity:
            raise ValueError(f'Missing fixed_cost or capacity for plant {p}')
        for c in customers:
            if c not in transport_cost[p]:
                raise ValueError(f'Missing transport cost for plant {p} to customer {c}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(plants, vtype=GRB.BINARY, name='')
    x = m.addVars([(p, c) for p in plants for c in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[p] * y[p] for p in plants)) + gurobipy.quicksum((transport_cost[p][c] * x[p, c] for p in plants for c in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gurobipy.quicksum((x[p, c] for p in plants)) == demand[c])
    for p in plants:
        m.addConstr(gurobipy.quicksum((x[p, c] for c in customers)) <= capacity[p] * y[p])
    m.optimize()
    return m
m = solve_problem()