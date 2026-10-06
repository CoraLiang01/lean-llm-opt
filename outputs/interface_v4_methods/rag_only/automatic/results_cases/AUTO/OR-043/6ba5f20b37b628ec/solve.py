CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A pharmacy chain needs to replenish its drug inventory. The ‘products.csv’ file provides a table of '
          'benefits associated with each drug product. There are overall stock capacity limits for pharmacy chains '
          'detailed in the ‘capacity.csv’ file.Our objective is to decide which drugs to order each day and in what '
          'quantities to maximise the overall benefit while adhering to the overall stock capacity. The decision '
          'variable x_i represents the number of units of the ith drug to be ordered each day.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '520'}}],
             'returned_rows': 1,
             'role': 'overall stock capacity parameter',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'NSAIDs', 'Value': '250', 'Weight': '913'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '178', 'Weight': '754'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '313', 'Weight': '428'}},
                         {'source_row': 3, 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '711'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Antipsychotics', 'Value': '934', 'Weight': '291'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Antihistamines', 'Value': '114', 'Weight': '302'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Corticosteroids', 'Value': '1357', 'Weight': '50'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Beta Blockers', 'Value': '156', 'Weight': '250'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '1780', 'Weight': '178'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Angiotensin II Receptor Blockers',
                                     'Value': '405',
                                     'Weight': '378'}},
                         {'source_row': 14, 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}},
                         {'source_row': 15, 'values': {'ProductName': 'Statins', 'Value': '320', 'Weight': '97'}},
                         {'source_row': 16, 'values': {'ProductName': 'Insulin', 'Value': '1357', 'Weight': '470'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Anticoagulants', 'Value': '1357', 'Weight': '341'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '405', 'Weight': '121'}},
                         {'source_row': 19, 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}],
             'returned_rows': 20,
             'role': 'drug product decision and benefit table',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    cap_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            cap_table = t
            break
    if cap_table is None:
        raise ValueError('Capacity table file_0_view_0 not found')
    if len(cap_table['records']) != 1:
        raise ValueError('Expected exactly one capacity record')
    try:
        C = int(cap_table['records'][0]['values']['Capacity'])
    except Exception:
        raise ValueError('Invalid or missing Capacity value')
    prod_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_1_view_0':
            prod_table = t
            break
    if prod_table is None:
        raise ValueError('Product table file_1_view_0 not found')
    products = []
    b = {}
    w = {}
    for rec in prod_table['records']:
        pname = rec['values']['ProductName']
        try:
            bval = float(rec['values']['Value'])
            wval = float(rec['values']['Weight'])
        except Exception:
            raise ValueError(f'Invalid Value or Weight for product {pname}')
        products.append(pname)
        b[pname] = bval
        w[pname] = wval
    for p in products:
        if p not in b or p not in w:
            raise ValueError(f'Missing benefit or weight for product {p}')
    m = gp.Model()
    x = m.addVars(products, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(gp.quicksum((b[p] * x[p] for p in products)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[p] * x[p] for p in products)) <= C, name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for p in products:
            print(f'{x[p].VarName} {x[p].X}')
    else:
        print(m.Status)
    return m
m = solve_problem(CSVQA_DATA)