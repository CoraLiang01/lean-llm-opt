CSVQA_DATA = {'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RestaurantSalesreport.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1483', 'Initial Inventory': '10440.0', 'Product Name': 'Aalopuri', 'Revenue': '20'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table['records']
    I = []
    r = {}
    d = {}
    s = {}
    for rec in records:
        values = rec['values']
        pname = values.get('Product Name', '')
        if not pname.startswith('Aalop'):
            continue
        if 'Revenue' not in values or 'Demand' not in values or 'Initial Inventory' not in values:
            raise ValueError(f'Missing parameter in record {rec}')
        try:
            r_val = float(values['Revenue'])
            d_val = float(values['Demand'])
            s_val = float(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {rec}') from e
        I.append(pname)
        r[pname] = r_val
        d[pname] = d_val
        s[pname] = s_val
    if not I:
        raise ValueError('No Aalop products found in the data')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
