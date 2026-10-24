CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'OnlineRetailSalesDataset.csv', 'records': [{'data_row_number': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    ids = []
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        vals = rec['values']
        id_num = vals['id_number']
        ids.append(id_num)
        rev[id_num] = float(vals['Revenue'])
        dem[id_num] = int(vals['Demand'])
        inv[id_num] = int(vals['Initial Inventory'])
    for i in ids:
        if i not in rev or i not in dem or i not in inv:
            raise ValueError(f'Missing parameter for {i}')
    x = m.addVars(ids, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    for i in ids:
        m.addConstr(x[i] <= dem[i], name=f'dem_{i}')
        m.addConstr(x[i] <= inv[i], name=f'inv_{i}')
    m.optimize()
    return m
m = solve_problem()
