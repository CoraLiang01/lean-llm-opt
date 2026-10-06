CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products with revenue data in the ‘Revenue’ column. The '
          'retailer aims to maximize total revenue using the initial inventory of products classified under ‘ELE-S’. '
          'Inventory levels are given in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘ELE-S’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesStoreoverview.csv',
             'filters': {'conditions': [{'column': 'Product_Reference',
                                         'dtype': 'string',
                                         'evidence': "'ELE-S' product i",
                                         'operator': 'prefix',
                                         'value': 'ELE-S'}],
                         'logic': 'and'},
             'original_rows': 161,
             'records': [{'source_row': 36,
                          'values': {'Demand': '295',
                                     'Initial Inventory': '2000.0',
                                     'Product_Reference': 'ELE-SMA-10000463',
                                     'Revenue': '4.0'}},
                         {'source_row': 37,
                          'values': {'Demand': '1002',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10000487',
                                     'Revenue': '14.0'}},
                         {'source_row': 38,
                          'values': {'Demand': '958',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10003333',
                                     'Revenue': '14.0'}},
                         {'source_row': 39,
                          'values': {'Demand': '777',
                                     'Initial Inventory': '6000.0',
                                     'Product_Reference': 'ELE-SMA-10009012',
                                     'Revenue': '4.0'}},
                         {'source_row': 40,
                          'values': {'Demand': '271',
                                     'Initial Inventory': '2000.0',
                                     'Product_Reference': 'ELE-SMA-10009999',
                                     'Revenue': '4.0'}},
                         {'source_row': 41,
                          'values': {'Demand': '244',
                                     'Initial Inventory': '2000.0',
                                     'Product_Reference': 'ELE-SMA-10011234',
                                     'Revenue': '4.0'}},
                         {'source_row': 42,
                          'values': {'Demand': '990',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10027456',
                                     'Revenue': '14.0'}},
                         {'source_row': 43,
                          'values': {'Demand': '1000',
                                     'Initial Inventory': '7000.0',
                                     'Product_Reference': 'ELE-SMA-10028567',
                                     'Revenue': '14.0'}},
                         {'source_row': 44,
                          'values': {'Demand': '169',
                                     'Initial Inventory': '1200.0',
                                     'Product_Reference': 'ELE-SPE-10000484',
                                     'Revenue': '2.4'}},
                         {'source_row': 45,
                          'values': {'Demand': '155',
                                     'Initial Inventory': '1200.0',
                                     'Product_Reference': 'ELE-SPE-10003030',
                                     'Revenue': '2.4'}},
                         {'source_row': 46,
                          'values': {'Demand': '327',
                                     'Initial Inventory': '2400.0',
                                     'Product_Reference': 'ELE-SPE-10024123',
                                     'Revenue': '2.4'}},
                         {'source_row': 47,
                          'values': {'Demand': '174',
                                     'Initial Inventory': '1200.0',
                                     'Product_Reference': 'ELE-SPE-10025234',
                                     'Revenue': '2.4'}}],
             'returned_rows': 12,
             'role': 'product revenue, demand, and inventory for ELE-S products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == table_id:
            table = t
            break
    if table is None:
        raise ValueError('Required table_id not found in CSVQA_DATA.')
    I = []
    r = {}
    s = {}
    d = {}
    for rec in table['records']:
        vals = rec['values']
        prod = vals['Product_Reference']
        I.append(prod)
        try:
            r[prod] = float(vals['Revenue'])
            s[prod] = float(vals['Initial Inventory'])
            d[prod] = float(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {prod}: {e}')
    for prod in I:
        if prod not in r or prod not in s or prod not in d:
            raise ValueError(f'Missing parameter for product {prod}')
    m = gp.Model()
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(gp.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= s[i], name='')
        m.addConstr(x[i] <= d[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_DATA)