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
             'role': 'equipment and product processing data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    products = ['Product I', 'Product II', 'Product III']
    product_ids = ['I', 'II', 'III']
    product_map = dict(zip(products, product_ids))
    A = ['A1', 'A2']
    B = ['B1', 'B2', 'B3']
    A_p = {'I': ['A1', 'A2'], 'II': ['A1', 'A2'], 'III': ['A2']}
    B_p = {'I': ['B1', 'B2', 'B3'], 'II': ['B1'], 'III': ['B2']}
    equip_rows = []
    for (idx, row) in frame.iterrows():
        equip = row['Equipment / Cost']
        if equip in A or equip in B:
            equip_rows.append((equip, row))
    t_ep = {}
    T_e = {}
    C_e = {}
    P_e = {}
    for (equip, row) in equip_rows:
        T_e[equip] = float(row['Available Equipment Operating Time'])
        C_e[equip] = float(row['Equipment Cost at Full Load (yuan)'])
        P_e[equip] = []
        for (prod, pid) in zip(products, product_ids):
            val = row[prod]
            if isinstance(val, str) and val.strip() != '':
                t_ep[equip, pid] = float(val)
                P_e[equip].append(pid)
    c_p = {}
    s_p = {}
    for (idx, row) in frame.iterrows():
        if row['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
            for (prod, pid) in zip(products, product_ids):
                val = row[prod]
                if isinstance(val, str) and val.strip() != '':
                    c_p[pid] = float(val)
        if row['Equipment / Cost'] == 'Unit Price (yuan/unit)':
            for (prod, pid) in zip(products, product_ids):
                val = row[prod]
                if isinstance(val, str) and val.strip() != '':
                    s_p[pid] = float(val)
    xA_keys = []
    for p in product_ids:
        for e in A_p[p]:
            if (e, p) in t_ep:
                xA_keys.append((e, p))
    xB_keys = []
    for p in product_ids:
        for e in B_p[p]:
            if (e, p) in t_ep:
                xB_keys.append((e, p))
    y_keys = list(product_ids)
    m = gp.Model('FactoryProduction')
    xA_vars = m.addVars(xA_keys, lb=0.0, name='')
    xB_vars = m.addVars(xB_keys, lb=0.0, name='')
    y_vars = m.addVars(y_keys, lb=0.0, name='')
    for p in product_ids:
        m.addConstr(gp.quicksum((xA_vars[e, p] for e in A_p[p] if (e, p) in xA_vars)) == y_vars[p], name=f'ConsistA_{p}')
        m.addConstr(gp.quicksum((xB_vars[e, p] for e in B_p[p] if (e, p) in xB_vars)) == y_vars[p], name=f'ConsistB_{p}')
    for e in A + B:
        if e in P_e and len(P_e[e]) > 0:
            expr = 0
            for p in P_e[e]:
                if e in A and (e, p) in xA_vars:
                    expr += t_ep[e, p] * xA_vars[e, p]
                if e in B and (e, p) in xB_vars:
                    expr += t_ep[e, p] * xB_vars[e, p]
            m.addConstr(expr <= T_e[e], name=f'Time_{e}')
    profit_expr = gp.quicksum(((s_p[p] - c_p[p]) * y_vars[p] for p in product_ids))
    equip_cost_expr = 0
    for e in A + B:
        if e in P_e and len(P_e[e]) > 0:
            for p in P_e[e]:
                if e in A and (e, p) in xA_vars:
                    equip_cost_expr += C_e[e] / T_e[e] * t_ep[e, p] * xA_vars[e, p]
                if e in B and (e, p) in xB_vars:
                    equip_cost_expr += C_e[e] / T_e[e] * t_ep[e, p] * xB_vars[e, p]
    m.setObjective(profit_expr - equip_cost_expr, gp.GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)