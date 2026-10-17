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
 'route': 'FLP',
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
    table = None
    for t in data['tables']:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found")
    records = table['records']
    equipment = []
    T_e = {}
    c_e = {}
    t_pe_prod = {}
    for rec in records:
        values = rec['values']
        equip = values['Equipment / Cost']
        if equip in ('Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)'):
            continue
        equipment.append(equip)
        T_val = values['Available Equipment Operating Time']
        if T_val == '':
            raise ValueError(f'Missing operating time for equipment {equip}')
        T_e[equip] = float(T_val)
        c_val = values['Equipment Cost at Full Load (yuan)']
        if c_val == '':
            raise ValueError(f'Missing equipment cost for {equip}')
        c_e[equip] = float(c_val)
        t_pe_prod[equip] = {}
        for col, prod in (('Product I', 'I'), ('Product II', 'II'), ('Product III', 'III')):
            val = values[col]
            if val != '':
                t_pe_prod[equip][prod] = float(val)
    products = ['I', 'II', 'III']
    r_p = {}
    p_p = {}
    for rec in records:
        values = rec['values']
        equip = values['Equipment / Cost']
        if equip == 'Raw Material Cost (yuan/unit)':
            for col, prod in (('Product I', 'I'), ('Product II', 'II'), ('Product III', 'III')):
                val = values[col]
                if val == '':
                    raise ValueError(f'Missing raw material cost for product {prod}')
                r_p[prod] = float(val)
        elif equip == 'Unit Price (yuan/unit)':
            for col, prod in (('Product I', 'I'), ('Product II', 'II'), ('Product III', 'III')):
                val = values[col]
                if val == '':
                    raise ValueError(f'Missing unit price for product {prod}')
                p_p[prod] = float(val)
    for prod in products:
        if prod not in r_p or prod not in p_p:
            raise ValueError(f'Missing cost or price for product {prod}')
    for equip in equipment:
        if equip not in T_e or equip not in c_e:
            raise ValueError(f'Missing parameters for equipment {equip}')
    M = sum((T_e[e] for e in equipment))
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(p, e) for p in products for e in equipment], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum(((p_p[p] - r_p[p]) * y[p] for p in products))
    obj -= gurobipy.quicksum((c_e[e] * t_pe_prod[e][p] * x[p, e] for e in equipment for p in products if p in t_pe_prod[e]))
    m.setObjective(obj, GRB.MAXIMIZE)
    A_equip = [e for e in equipment if e.startswith('A')]
    B_equip = [e for e in equipment if e.startswith('B')]
    for p in products:
        m.addConstr(gurobipy.quicksum((x[p, e] for e in A_equip)) == y[p], name='')
        m.addConstr(gurobipy.quicksum((x[p, e] for e in B_equip)) == y[p], name='')
    for e in equipment:
        m.addConstr(gurobipy.quicksum((t_pe_prod[e][p] * x[p, e] for p in products if p in t_pe_prod[e])) <= T_e[e], name='')
    for e in equipment:
        for p in products:
            if p in t_pe_prod[e]:
                m.addConstr(x[p, e] <= M, name='')
            else:
                m.addConstr(x[p, e] == 0, name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')