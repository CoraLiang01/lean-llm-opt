CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket manager needs to select a variety of products to stock in different sections of the store. '
          'Particularly, the store has several sections, each with a display space limit provided in "capacity.csv." '
          'The predefined price and shelf space requirement of each product are detailed in "products.csv." The '
          'objective is to determine the optimal number of units of each product to stock in each section to maximize '
          'the total revenue, while ensuring that the total space used by the products in each section does not exceed '
          'the available capacity. The decision variables x_ij denote the number of units of product j to be placed in '
          'section i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['SectionID', 'archive_revision_number', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '100', 'SectionID': '1', 'archive_revision_number': '3'}},
                         {'source_row': 1,
                          'values': {'Capacity': '150', 'SectionID': '2', 'archive_revision_number': '7'}},
                         {'source_row': 2,
                          'values': {'Capacity': '120', 'SectionID': '3', 'archive_revision_number': '3'}},
                         {'source_row': 3,
                          'values': {'Capacity': '130', 'SectionID': '4', 'archive_revision_number': '4'}},
                         {'source_row': 4,
                          'values': {'Capacity': '90', 'SectionID': '5', 'archive_revision_number': '6'}},
                         {'source_row': 5,
                          'values': {'Capacity': '110', 'SectionID': '6', 'archive_revision_number': '4'}},
                         {'source_row': 6,
                          'values': {'Capacity': '160', 'SectionID': '7', 'archive_revision_number': '3'}},
                         {'source_row': 7,
                          'values': {'Capacity': '140', 'SectionID': '8', 'archive_revision_number': '8'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['record_keeper_group', 'ProductName', 'Value', 'archive_revision_number', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'ProductName': '1',
                                     'Value': '10',
                                     'Weight': '2',
                                     'archive_revision_number': '1',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 1,
                          'values': {'ProductName': '2',
                                     'Value': '15',
                                     'Weight': '3',
                                     'archive_revision_number': '4',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 2,
                          'values': {'ProductName': '3',
                                     'Value': '8',
                                     'Weight': '1',
                                     'archive_revision_number': '2',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 3,
                          'values': {'ProductName': '4',
                                     'Value': '12',
                                     'Weight': '2',
                                     'archive_revision_number': '9',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 4,
                          'values': {'ProductName': '5',
                                     'Value': '20',
                                     'Weight': '4',
                                     'archive_revision_number': '3',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 5,
                          'values': {'ProductName': '6',
                                     'Value': '25',
                                     'Weight': '5',
                                     'archive_revision_number': '7',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 6,
                          'values': {'ProductName': '7',
                                     'Value': '5',
                                     'Weight': '1',
                                     'archive_revision_number': '2',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 7,
                          'values': {'ProductName': '8',
                                     'Value': '30',
                                     'Weight': '6',
                                     'archive_revision_number': '3',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 8,
                          'values': {'ProductName': '9',
                                     'Value': '18',
                                     'Weight': '3',
                                     'archive_revision_number': '7',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 9,
                          'values': {'ProductName': '10',
                                     'Value': '22',
                                     'Weight': '4',
                                     'archive_revision_number': '5',
                                     'record_keeper_group': 'Team A'}}],
             'returned_rows': 10,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'SectionID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'SectionID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'SectionID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'SectionID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    section_df = CSVQA_FRAMES['file_0_view_0']
    product_df = CSVQA_FRAMES['file_1_view_0']
    S = section_df['SectionID'].tolist()
    P = product_df['ProductName'].tolist()
    c_s = {}
    for (_, row) in section_df.iterrows():
        sid = row['SectionID']
        cap = row['Capacity']
        if cap == '' or sid == '':
            raise ValueError(f'Missing Capacity or SectionID in section data: {row}')
        c_s[sid] = int(cap)
    v_p = {}
    w_p = {}
    for (_, row) in product_df.iterrows():
        pid = row['ProductName']
        val = row['Value']
        wt = row['Weight']
        if val == '' or wt == '' or pid == '':
            raise ValueError(f'Missing Value, Weight, or ProductName in product data: {row}')
        v_p[pid] = int(val)
        w_p[pid] = int(wt)
    if set(S) != set(c_s.keys()):
        raise ValueError('SectionID mismatch between S and c_s')
    if set(P) != set(v_p.keys()) or set(P) != set(w_p.keys()):
        raise ValueError('ProductName mismatch between P, v_p, and w_p')
    m = gp.Model('Supermarket_Section_Stocking')
    quantity_vars = m.addVars(S, P, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * quantity_vars[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_p[p] * quantity_vars[s, p] for p in P)) <= c_s[s] for s in S), name='')
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