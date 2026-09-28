CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A pharmacy chain needs to replenish its drug inventory. The ‘products.csv’ file provides a table of '
          'benefits associated with each drug product. There are overall stock capacity limits for pharmacy chains '
          'detailed in the ‘capacity.csv’ file.Our objective is to decide which drugs to order each day and in what '
          'quantities to maximise the overall benefit while adhering to the overall stock capacity. The decision '
          'variable x_i represents the number of units of the ith drug to be ordered each day.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '520'}}],
             'returned_rows': 1,
             'role': 'overall stock capacity',
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
             'role': 'drug product benefit and attributes',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'A pharmacy chain needs to replenish its drug inventory. The ‘products.csv’ file provides a table of benefits associated with each drug product. There are overall stock capacity limits for pharmacy chains detailed in the ‘capacity.csv’ file.Our objective is to decide which drugs to order each day and in what quantities to maximise the overall benefit while adhering to the overall stock capacity. The decision variable x_i represents the number of units of the ith drug to be ordered each day.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['Capacity'], 'file_index': 0, 'file_name': 'capacity.csv', 'filters': {}, 'original_rows': 1, 'records': [{'source_row': 0, 'values': {'Capacity': '520'}}], 'returned_rows': 1, 'role': 'overall stock capacity', 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_index': 1, 'file_name': 'products.csv', 'filters': {}, 'original_rows': 20, 'records': [{'source_row': 0, 'values': {'ProductName': 'NSAIDs', 'Value': '250', 'Weight': '913'}}, {'source_row': 1, 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '178', 'Weight': '754'}}, {'source_row': 2, 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '313', 'Weight': '428'}}, {'source_row': 3, 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '711'}}, {'source_row': 4, 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source_row': 5, 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source_row': 6, 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source_row': 7, 'values': {'ProductName': 'Antipsychotics', 'Value': '934', 'Weight': '291'}}, {'source_row': 8, 'values': {'ProductName': 'Antihistamines', 'Value': '114', 'Weight': '302'}}, {'source_row': 9, 'values': {'ProductName': 'Corticosteroids', 'Value': '1357', 'Weight': '50'}}, {'source_row': 10, 'values': {'ProductName': 'Beta Blockers', 'Value': '156', 'Weight': '250'}}, {'source_row': 11, 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '1780', 'Weight': '178'}}, {'source_row': 12, 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source_row': 13, 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source_row': 14, 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source_row': 15, 'values': {'ProductName': 'Statins', 'Value': '320', 'Weight': '97'}}, {'source_row': 16, 'values': {'ProductName': 'Insulin', 'Value': '1357', 'Weight': '470'}}, {'source_row': 17, 'values': {'ProductName': 'Anticoagulants', 'Value': '1357', 'Weight': '341'}}, {'source_row': 18, 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '405', 'Weight': '121'}}, {'source_row': 19, 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}], 'returned_rows': 20, 'role': 'drug product benefit and attributes', 'table_id': 'file_1_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
capacity_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        capacity_table = t
        break
if capacity_table is None:
    raise ValueError('Capacity table not found.')
if len(capacity_table['records']) != 1:
    raise ValueError('Expected exactly one capacity record.')
try:
    C = int(capacity_table['records'][0]['values']['Capacity'])
except Exception:
    raise ValueError('Capacity value missing or not an integer.')
products_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_1_view_0':
        products_table = t
        break
if products_table is None:
    raise ValueError('Products table not found.')
I = []
v = {}
w = {}
for rec in products_table['records']:
    vals = rec['values']
    pname = vals['ProductName']
    try:
        vi = int(vals['Value'])
        wi = int(vals['Weight'])
    except Exception:
        raise ValueError(f'Value or Weight missing or not integer for product {pname}.')
    I.append(pname)
    v[pname] = vi
    w[pname] = wi
if set(v.keys()) != set(I) or set(w.keys()) != set(I):
    raise ValueError('Mismatch in product indices for value/weight.')
m = gp.Model('Drug_Stock_Optimization')
x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')