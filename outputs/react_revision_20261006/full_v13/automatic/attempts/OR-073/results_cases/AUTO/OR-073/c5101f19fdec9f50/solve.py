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
             'role': 'equipment and product parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    A_equipment = []
    B_equipment = []
    all_equipment = []
    equipment_rows = {}
    for (idx, row) in frame.iterrows():
        eq = row['Equipment / Cost']
        if eq in ['A1', 'A2']:
            A_equipment.append(eq)
            all_equipment.append(eq)
            equipment_rows[eq] = idx
        elif eq in ['B1', 'B2', 'B3']:
            B_equipment.append(eq)
            all_equipment.append(eq)
            equipment_rows[eq] = idx
    products = ['Product I', 'Product II', 'Product III']
    c_p = {}
    s_p = {}
    for (idx, row) in frame.iterrows():
        if row['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
            for p in products:
                c_p[p] = float(row[p])
        if row['Equipment / Cost'] == 'Unit Price (yuan/unit)':
            for p in products:
                s_p[p] = float(row[p])
    t_ep = {}
    T_e = {}
    C_e = {}
    for eq in all_equipment:
        idx = equipment_rows[eq]
        row = frame.loc[idx]
        T_e[eq] = float(row['Available Equipment Operating Time'])
        C_e[eq] = float(row['Equipment Cost at Full Load (yuan)'])
        for p in products:
            val = row[p]
            if val != '' and val is not None:
                t_ep[eq, p] = float(val)
    xA_keys = []
    if ('A1', 'Product I') in t_ep:
        xA_keys.append(('A1', 'Product I'))
    if ('A2', 'Product I') in t_ep:
        xA_keys.append(('A2', 'Product I'))
    if ('A1', 'Product II') in t_ep:
        xA_keys.append(('A1', 'Product II'))
    if ('A2', 'Product II') in t_ep:
        xA_keys.append(('A2', 'Product II'))
    if ('A2', 'Product III') in t_ep:
        xA_keys.append(('A2', 'Product III'))
    xB_keys = []
    if ('B1', 'Product I') in t_ep:
        xB_keys.append(('B1', 'Product I'))
    if ('B2', 'Product I') in t_ep:
        xB_keys.append(('B2', 'Product I'))
    if ('B3', 'Product I') in t_ep:
        xB_keys.append(('B3', 'Product I'))
    if ('B1', 'Product II') in t_ep:
        xB_keys.append(('B1', 'Product II'))
    if ('B2', 'Product III') in t_ep:
        xB_keys.append(('B2', 'Product III'))
    m = gp.Model('FactoryProduction')
    m.Params.MIPGap = 0.0001
    xA_vars = m.addVars(xA_keys, lb=0.0, name='')
    xB_vars = m.addVars(xB_keys, lb=0.0, name='')
    y_vars = m.addVars(products, lb=0.0, name='')
    m.addConstr(xA_vars.get(('A1', 'Product I'), 0) + xA_vars.get(('A2', 'Product I'), 0) == y_vars['Product I'], name='flowA_I')
    m.addConstr(xB_vars.get(('B1', 'Product I'), 0) + xB_vars.get(('B2', 'Product I'), 0) + xB_vars.get(('B3', 'Product I'), 0) == y_vars['Product I'], name='flowB_I')
    m.addConstr(xA_vars.get(('A1', 'Product II'), 0) + xA_vars.get(('A2', 'Product II'), 0) == y_vars['Product II'], name='flowA_II')
    m.addConstr(xB_vars.get(('B1', 'Product II'), 0) == y_vars['Product II'], name='flowB_II')
    m.addConstr(xA_vars.get(('A2', 'Product III'), 0) == y_vars['Product III'], name='flowA_III')
    m.addConstr(xB_vars.get(('B2', 'Product III'), 0) == y_vars['Product III'], name='flowB_III')
    m.addConstr(gp.quicksum((t_ep['A1', p] * xA_vars['A1', p] for p in ['Product I', 'Product II'] if ('A1', p) in xA_vars)) <= T_e['A1'], name='cap_A1')
    m.addConstr(gp.quicksum((t_ep['A2', p] * xA_vars['A2', p] for p in ['Product I', 'Product II', 'Product III'] if ('A2', p) in xA_vars)) <= T_e['A2'], name='cap_A2')
    m.addConstr(gp.quicksum((t_ep['B1', p] * xB_vars['B1', p] for p in ['Product I', 'Product II'] if ('B1', p) in xB_vars)) <= T_e['B1'], name='cap_B1')
    m.addConstr(gp.quicksum((t_ep['B2', p] * xB_vars['B2', p] for p in ['Product I', 'Product III'] if ('B2', p) in xB_vars)) <= T_e['B2'], name='cap_B2')
    m.addConstr(gp.quicksum((t_ep['B3', p] * xB_vars['B3', p] for p in ['Product I'] if ('B3', p) in xB_vars)) <= T_e['B3'], name='cap_B3')
    revenue = gp.quicksum((s_p[p] * y_vars[p] for p in products))
    raw_cost = gp.quicksum((c_p[p] * y_vars[p] for p in products))
    equip_cost = 0
    for e in A_equipment:
        equip_cost += C_e[e] / T_e[e] * gp.quicksum((t_ep[e, p] * xA_vars[e, p] for p in products if (e, p) in xA_vars))
    for e in B_equipment:
        equip_cost += C_e[e] / T_e[e] * gp.quicksum((t_ep[e, p] * xB_vars[e, p] for p in products if (e, p) in xB_vars))
    m.setObjective(revenue - raw_cost - equip_cost, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.status}')