CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A chain pharmacy needs to restock its drug inventory. For each drug type (e.g., NSAIDs, antirheumatic '
          'drugs, acetic-acid derivatives, etc.), the benefit coefficient is listed in “products.csv.” The pharmacy '
          'faces a single overall inventory-capacity constraint, provided in “capacity.csv.”\n'
          '    The goal is to decide the daily order quantity of each drug typeso as to maximize total benefit while '
          'ensuring that the total weight of all ordered units does not exceed the overall capacity.The decision '
          'variables x_i  represent the number of units of type  i  drug to be ordered daily.The decision variables '
          'must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '4120'}}],
             'returned_rows': 1,
             'role': 'overall inventory capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 20,
             'records': [{'source_row': 0, 'values': {'ProductName': 'NSAIDs', 'Value': '585', 'Weight': '50'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '557', 'Weight': '329'}},
                         {'source_row': 2,
                          'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '963', 'Weight': '410'}},
                         {'source_row': 3, 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '452'}},
                         {'source_row': 4,
                          'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}},
                         {'source_row': 5,
                          'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}},
                         {'source_row': 7,
                          'values': {'ProductName': 'Antipsychotics', 'Value': '461', 'Weight': '291'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Antihistamines', 'Value': '840', 'Weight': '302'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Corticosteroids', 'Value': '999', 'Weight': '50'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Beta Blockers', 'Value': '392', 'Weight': '250'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '874', 'Weight': '178'}},
                         {'source_row': 12,
                          'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}},
                         {'source_row': 13,
                          'values': {'ProductName': 'Angiotensin II Receptor Blockers',
                                     'Value': '405',
                                     'Weight': '378'}},
                         {'source_row': 14, 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}},
                         {'source_row': 15, 'values': {'ProductName': 'Statins', 'Value': '913', 'Weight': '97'}},
                         {'source_row': 16, 'values': {'ProductName': 'Insulin', 'Value': '754', 'Weight': '470'}},
                         {'source_row': 17,
                          'values': {'ProductName': 'Anticoagulants', 'Value': '428', 'Weight': '341'}},
                         {'source_row': 18,
                          'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '711', 'Weight': '121'}},
                         {'source_row': 19, 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}],
             'returned_rows': 20,
             'role': 'drug type decision and coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_frame = CSVQA_FRAMES['file_1_view_0']
    capacity_frame = CSVQA_FRAMES['file_0_view_0']
    I = []
    v = {}
    w = {}
    for (source_row, row) in products_frame.iterrows():
        product = row['ProductName']
        I.append(product)
        v[product] = float(row['Value'])
        w[product] = float(row['Weight'])
    if capacity_frame.shape[0] != 1:
        raise ValueError('Expected exactly one row in capacity.csv for total capacity.')
    C = float(capacity_frame.iloc[0]['Capacity'])
    m = gp.Model('pharmacy_inventory')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * quantity_vars[i] for i in I)) <= C)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')