CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A '
          'and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of '
          'equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A '
          'equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when '
          'completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 '
          'and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment '
          'operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine '
          'the optimal production plan to maximize profit. All production quantities should be continuous.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Equipment / Cost',
                         'Product I',
                         'Product II',
                         'Product III',
                         'Available Equipment Operating Time',
                         'Equipment Cost at Full Load (yuan)'],
             'file_index': 0,
             'file_name': '43.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0,
                          'values': {'Available Equipment Operating Time': '6000',
                                     'Equipment / Cost': 'A1',
                                     'Equipment Cost at Full Load (yuan)': '300',
                                     'Product I': '5',
                                     'Product II': '10',
                                     'Product III': ''}},
                         {'source_row': 1,
                          'values': {'Available Equipment Operating Time': '10000',
                                     'Equipment / Cost': 'A2',
                                     'Equipment Cost at Full Load (yuan)': '321',
                                     'Product I': '7',
                                     'Product II': '9',
                                     'Product III': '12'}},
                         {'source_row': 2,
                          'values': {'Available Equipment Operating Time': '8000',
                                     'Equipment / Cost': 'A3',
                                     'Equipment Cost at Full Load (yuan)': '203',
                                     'Product I': '6',
                                     'Product II': '11',
                                     'Product III': '2'}},
                         {'source_row': 3,
                          'values': {'Available Equipment Operating Time': '4000',
                                     'Equipment / Cost': 'B1',
                                     'Equipment Cost at Full Load (yuan)': '250',
                                     'Product I': '6',
                                     'Product II': '8',
                                     'Product III': ''}},
                         {'source_row': 4,
                          'values': {'Available Equipment Operating Time': '7000',
                                     'Equipment / Cost': 'B2',
                                     'Equipment Cost at Full Load (yuan)': '783',
                                     'Product I': '4',
                                     'Product II': '',
                                     'Product III': '11'}},
                         {'source_row': 5,
                          'values': {'Available Equipment Operating Time': '4000',
                                     'Equipment / Cost': 'B3',
                                     'Equipment Cost at Full Load (yuan)': '200',
                                     'Product I': '7',
                                     'Product II': '',
                                     'Product III': ''}},
                         {'source_row': 6,
                          'values': {'Available Equipment Operating Time': '5000',
                                     'Equipment / Cost': 'B4',
                                     'Equipment Cost at Full Load (yuan)': '300',
                                     'Product I': '3',
                                     'Product II': '5',
                                     'Product III': '8'}},
                         {'source_row': 7,
                          'values': {'Available Equipment Operating Time': '',
                                     'Equipment / Cost': 'Raw Material Cost (yuan/unit)',
                                     'Equipment Cost at Full Load (yuan)': '',
                                     'Product I': '0.25',
                                     'Product II': '0.35',
                                     'Product III': '0.5'}},
                         {'source_row': 8,
                          'values': {'Available Equipment Operating Time': '',
                                     'Equipment / Cost': 'Unit Price (yuan/unit)',
                                     'Equipment Cost at Full Load (yuan)': '',
                                     'Product I': '1.25',
                                     'Product II': '2',
                                     'Product III': '2.8'}}],
             'returned_rows': 9,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = data['tables'][0]
    records = table['records']
    E = {'A1', 'A2', 'B1', 'B2', 'B3'}
    P = {'I', 'II', 'III'}
    L = {('A1', 'I'), ('A1', 'II'), ('A2', 'I'), ('A2', 'II'), ('A2', 'III'), ('B1', 'I'), ('B1', 'II'), ('B2', 'I'), ('B2', 'III'), ('B3', 'I')}
    a = {}
    c = {}
    t = {}
    raw = {}
    price = {}
    for rec in records:
        vals = rec['values']
        e_name = vals['Equipment / Cost']
        if e_name in E:
            a_val = vals['Available Equipment Operating Time']
            c_val = vals['Equipment Cost at Full Load (yuan)']
            if a_val == '' or c_val == '':
                continue
            a[e_name] = float(a_val)
            c[e_name] = float(c_val)
            t[e_name] = {}
            for p in P:
                prod_val = vals[f'Product {p}']
                if prod_val != '':
                    t[e_name][p] = float(prod_val)
        elif e_name == 'Raw Material Cost (yuan/unit)':
            for p in P:
                prod_val = vals[f'Product {p}']
                if prod_val != '':
                    raw[p] = float(prod_val)
        elif e_name == 'Unit Price (yuan/unit)':
            for p in P:
                prod_val = vals[f'Product {p}']
                if prod_val != '':
                    price[p] = float(prod_val)
    for e in E:
        if e not in a:
            raise ValueError(f'Missing available time for equipment {e}')
        if e not in c:
            raise ValueError(f'Missing cost for equipment {e}')
        if e not in t:
            raise ValueError(f'Missing processing times for equipment {e}')
        for p in P:
            if p not in t[e]:
                raise ValueError(f'Missing processing time for equipment {e} product {p}')
    for p in P:
        if p not in raw:
            raise ValueError(f'Missing raw material cost for product {p}')
        if p not in price:
            raise ValueError(f'Missing unit price for product {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(L, lb=0, vtype=GRB.CONTINUOUS, name='')
    profit_expr = gurobipy.quicksum(((price[p] - raw[p]) * x[e, p] for (e, p) in L)) - gurobipy.quicksum((c[e] * t[e][p] * x[e, p] for (e, p) in L))
    m.setObjective(profit_expr, GRB.MAXIMIZE)
    for e in E:
        m.addConstr(gurobipy.quicksum((t[e][p] * x[e, p] for p in P if (e, p) in L)) <= a[e])
    m.optimize()
    return m
m = solve_problem()