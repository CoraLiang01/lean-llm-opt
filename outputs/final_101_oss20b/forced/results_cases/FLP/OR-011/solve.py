CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'OnlineRetailSalesDataset.csv', 'records': [{'data_row_number': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model('MaxRevenue')
    m.setParam('MIPGap', 1e-06)
    table = next((t for t in CSVQA_DATA['tables'] if t.get('table_id') == 'file_0_view_0'))
    I = []
    r = {}
    d = {}
    v = {}
    for rec in table['records']:
        vals = rec['values']
        id_num = vals.get('id_number')
        if id_num is None:
            continue
        if not str(id_num).startswith('id999'):
            continue
        I.append(id_num)
        rev_str = vals.get('Revenue')
        if rev_str is None:
            raise ValueError(f'Missing Revenue for {id_num}')
        try:
            rev = float(rev_str)
        except Exception as exc:
            raise ValueError(f'Invalid Revenue for {id_num}') from exc
        r[id_num] = rev
        dem_str = vals.get('Demand')
        if dem_str is None:
            raise ValueError(f'Missing Demand for {id_num}')
        try:
            dem = int(dem_str)
        except Exception as exc:
            raise ValueError(f'Invalid Demand for {id_num}') from exc
        d[id_num] = dem
        inv_str = vals.get('Initial Inventory')
        if inv_str is None:
            raise ValueError(f'Missing Initial Inventory for {id_num}')
        try:
            inv = int(inv_str)
        except Exception as exc:
            raise ValueError(f'Invalid Initial Inventory for {id_num}') from exc
        v[id_num] = inv
    ub_dict = {i: min(d[i], v[i]) for i in I}
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=ub_dict, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name=f'demand_{i}')
        m.addConstr(x[i] <= v[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()
