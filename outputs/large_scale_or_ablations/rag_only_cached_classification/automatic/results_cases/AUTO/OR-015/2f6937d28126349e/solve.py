CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for '
          'these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant '
          'aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, '
          'which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, '
          'and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to '
          'be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The '
          'variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'Aalop' products",
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Aalop'}],
                         'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB

def solve_abstract_optimization_model(CSVQA_DATA):
    table_id = 'file_0_view_0'
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    if table_id not in tables:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA.')
    table = tables[table_id]
    records = table['records']
    I = []
    r = {}
    d = {}
    s = {}
    prefix = 'Aalop'
    for rec in records:
        vals = rec['values']
        pname = vals['Product Name']
        if not re.match(f'^{re.escape(prefix)}', pname):
            continue
        I.append(pname)
        try:
            r_i = float(vals['Revenue'])
            d_i = float(vals['Demand'])
            s_i = float(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pname}: {e}')
        r[pname] = r_i
        d[pname] = d_i
        s[pname] = s_i
    if not I:
        raise ValueError("No products with prefix 'Aalop' found.")
    for pname in I:
        if pname not in r or pname not in d or pname not in s:
            raise ValueError(f'Missing data for product {pname}.')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub={i: min(int(d[i]), int(s[i])) for i in I}, name='')
    m.setObjective(sum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= s[i], name='')
        m.addConstr(x[i] <= d[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName}: {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_abstract_optimization_model(CSVQA_DATA)