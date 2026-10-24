CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'OnlineRetailSalesDataset.csv', 'records': [{'data_row_number': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    I = []
    r_i = {}
    d_i = {}
    s_i = {}
    for rec in records:
        vals = rec.get('values', {})
        id_num = vals.get('id_number')
        if id_num is None:
            raise ValueError("Record missing 'id_number'")
        if not str(id_num).startswith('id999'):
            continue
        I.append(id_num)
        try:
            revenue = float(vals.get('Revenue', 0))
            demand = int(vals.get('Demand', 0))
            inventory = int(vals.get('Initial Inventory', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f'Invalid numeric value in record {rec}: {e}')
        r_i[id_num] = revenue
        d_i[id_num] = demand
        s_i[id_num] = inventory
    if not I:
        raise ValueError("No records with id starting with 'id999' found")
    for i in I:
        if i not in r_i or i not in d_i or i not in s_i:
            raise ValueError(f'Missing parameter for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d_i[i])
        m.addConstr(x[i] <= s_i[i])
    m.optimize()
    return m
m = solve_problem()
