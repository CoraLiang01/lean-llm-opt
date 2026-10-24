CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesdata.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '3066513', 'Initial Inventory': '22749210', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    I = [rec['data_row_number'] for rec in records if rec['values']['Product Name'].startswith('Baby')]
    r = {}
    d = {}
    s = {}
    for rec in records:
        i = rec['data_row_number']
        if not rec['values']['Product Name'].startswith('Baby'):
            continue
        r[i] = float(rec['values']['Revenue'])
        d[i] = int(rec['values']['Demand'])
        s[i] = int(rec['values']['Initial Inventory'])
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name='')
        m.addConstr(x[i] <= s[i], name='')
    m.optimize()
    return m
m = solve_problem()
