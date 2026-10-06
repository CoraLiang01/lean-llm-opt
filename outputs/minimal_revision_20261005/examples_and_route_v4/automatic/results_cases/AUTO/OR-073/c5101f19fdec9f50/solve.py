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
 'route': 'RA',
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
             'role': 'equipment and product parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA['tables'][0]['records']
    products = ['Product I', 'Product II', 'Product III']
    E_A = ['A1', 'A2']
    E_B = ['B1', 'B2', 'B3']
    E_A_p = {'Product I': ['A1', 'A2'], 'Product II': ['A1', 'A2'], 'Product III': ['A2']}
    E_B_p = {'Product I': ['B1', 'B2', 'B3'], 'Product II': ['B1'], 'Product III': ['B2']}
    P_A_e = {e: [] for e in E_A}
    for p in products:
        for e in E_A_p[p]:
            P_A_e[e].append(p)
    P_B_e = {e: [] for e in E_B}
    for p in products:
        for e in E_B_p[p]:
            P_B_e[e].append(p)
    t = {}
    T = {}
    C = {}
    for rec in data:
        eq = rec['values']['Equipment / Cost']
        if eq in E_A + E_B:
            for p in products:
                val = rec['values'][p]
                if val != '' and val is not None:
                    t[eq, p] = float(val)
            T[eq] = float(rec['values']['Available Equipment Operating Time'])
            C[eq] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
    for rec in data:
        eq = rec['values']['Equipment / Cost']
        if eq == 'Raw Material Cost (yuan/unit)':
            c_raw = {p: float(rec['values'][p]) for p in products}
        if eq == 'Unit Price (yuan/unit)':
            r = {p: float(rec['values'][p]) for p in products}
    m = gp.Model('factory_production')
    m.Params.MIPGap = 0.0001
    xA_keys = []
    for e in E_A:
        for p in P_A_e[e]:
            xA_keys.append((e, p))
    xB_keys = []
    for e in E_B:
        for p in P_B_e[e]:
            xB_keys.append((e, p))
    xA = m.addVars(xA_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    xB = m.addVars(xB_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    u = m.addVars(E_A + E_B, lb=0, vtype=GRB.CONTINUOUS, name='')
    for p in products:
        m.addConstr(y[p] == gp.quicksum((xA[e, p] for e in E_A_p[p] if (e, p) in xA)))
        m.addConstr(y[p] == gp.quicksum((xB[e, p] for e in E_B_p[p] if (e, p) in xB)))
    for e in E_A:
        expr = gp.quicksum((t[e, p] * xA[e, p] for p in P_A_e[e] if (e, p) in xA))
        m.addConstr(expr <= T[e])
        m.addConstr(u[e] == expr)
    for e in E_B:
        expr = gp.quicksum((t[e, p] * xB[e, p] for p in P_B_e[e] if (e, p) in xB))
        m.addConstr(expr <= T[e])
        m.addConstr(u[e] == expr)
    profit = gp.quicksum((r[p] * y[p] for p in products)) - gp.quicksum((c_raw[p] * y[p] for p in products)) - gp.quicksum((C[e] * u[e] / T[e] for e in E_A + E_B))
    m.setObjective(profit, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')