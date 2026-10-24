CSVQA_DATA = {'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'RestaurantSalesreport.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '1483', 'Initial Inventory': '10440.0', 'Product Name': 'Aalopuri', 'Revenue': '20'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = None
    for tbl in data['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    r = {}
    d = {}
    s = {}
    for rec in table.get('records', []):
        vals = rec.get('values', {})
        prod = vals.get('Product Name')
        if prod is None:
            continue
        try:
            rev = float(vals.get('Revenue', 0))
            dem = float(vals.get('Demand', 0))
            inv = float(vals.get('Initial Inventory', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Non-numeric value in record {rec['data_row_number']}: {e}")
        r[prod] = rev
        d[prod] = dem
        s[prod] = inv
    I = set(r.keys())
    for i in I:
        if i not in d or i not in s:
            raise ValueError(f'Missing parameter for product {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    for i in I:
        m.addConstr(x[i] <= d[i], name='')
        m.addConstr(x[i] <= s[i], name='')
    obj = gurobipy.quicksum((r[i] * x[i] for i in I))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
