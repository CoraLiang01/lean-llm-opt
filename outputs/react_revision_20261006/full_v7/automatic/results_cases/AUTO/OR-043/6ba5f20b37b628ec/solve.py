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
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '520'}}],
             'returned_rows': 1,
             'role': 'overall stock capacity parameter',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    products_df = CSVQA_FRAMES['file_1_view_0']
    capacity_df = CSVQA_FRAMES['file_0_view_0']
    products = list(products_df['ProductName'])
    try:
        v = {row['ProductName']: float(row['Value']) for (_, row) in products_df.iterrows()}
        w = {row['ProductName']: float(row['Weight']) for (_, row) in products_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Value or Weight columns: {e}')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception as e:
        raise ValueError(f'Error parsing Capacity: {e}')
    if set(products) != set(v.keys()) or set(products) != set(w.keys()):
        raise ValueError('Mismatch in product identifiers between products and parameters.')
    m = gp.Model('pharmacy_inventory_optimization')
    quantity_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * quantity_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * quantity_vars[i] for i in products)) <= C, name='capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)