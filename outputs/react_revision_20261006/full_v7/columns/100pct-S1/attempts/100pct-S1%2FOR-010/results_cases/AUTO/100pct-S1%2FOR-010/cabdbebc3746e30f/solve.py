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
 'tables': [{'columns': ['SectionID', 'archived_attachment_count', 'archive_revision_number', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '100',
                                     'SectionID': '1',
                                     'archive_revision_number': '3',
                                     'archived_attachment_count': '1'}},
                         {'source_row': 1,
                          'values': {'Capacity': '150',
                                     'SectionID': '2',
                                     'archive_revision_number': '7',
                                     'archived_attachment_count': '1'}},
                         {'source_row': 2,
                          'values': {'Capacity': '120',
                                     'SectionID': '3',
                                     'archive_revision_number': '3',
                                     'archived_attachment_count': '3'}},
                         {'source_row': 3,
                          'values': {'Capacity': '130',
                                     'SectionID': '4',
                                     'archive_revision_number': '4',
                                     'archived_attachment_count': '2'}},
                         {'source_row': 4,
                          'values': {'Capacity': '90',
                                     'SectionID': '5',
                                     'archive_revision_number': '6',
                                     'archived_attachment_count': '3'}},
                         {'source_row': 5,
                          'values': {'Capacity': '110',
                                     'SectionID': '6',
                                     'archive_revision_number': '4',
                                     'archived_attachment_count': '3'}},
                         {'source_row': 6,
                          'values': {'Capacity': '160',
                                     'SectionID': '7',
                                     'archive_revision_number': '3',
                                     'archived_attachment_count': '1'}},
                         {'source_row': 7,
                          'values': {'Capacity': '140',
                                     'SectionID': '8',
                                     'archive_revision_number': '8',
                                     'archived_attachment_count': '3'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['record_keeper_group',
                         'ProductName',
                         'Value',
                         'archive_revision_number',
                         'Weight',
                         'archived_attachment_count'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'ProductName': '1',
                                     'Value': '10',
                                     'Weight': '2',
                                     'archive_revision_number': '1',
                                     'archived_attachment_count': '4',
                                     'record_keeper_group': 'Team B'}},
                         {'source_row': 1,
                          'values': {'ProductName': '2',
                                     'Value': '15',
                                     'Weight': '3',
                                     'archive_revision_number': '4',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 2,
                          'values': {'ProductName': '3',
                                     'Value': '8',
                                     'Weight': '1',
                                     'archive_revision_number': '2',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 3,
                          'values': {'ProductName': '4',
                                     'Value': '12',
                                     'Weight': '2',
                                     'archive_revision_number': '9',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 4,
                          'values': {'ProductName': '5',
                                     'Value': '20',
                                     'Weight': '4',
                                     'archive_revision_number': '3',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 5,
                          'values': {'ProductName': '6',
                                     'Value': '25',
                                     'Weight': '5',
                                     'archive_revision_number': '7',
                                     'archived_attachment_count': '2',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 6,
                          'values': {'ProductName': '7',
                                     'Value': '5',
                                     'Weight': '1',
                                     'archive_revision_number': '2',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team A'}},
                         {'source_row': 7,
                          'values': {'ProductName': '8',
                                     'Value': '30',
                                     'Weight': '6',
                                     'archive_revision_number': '3',
                                     'archived_attachment_count': '3',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 8,
                          'values': {'ProductName': '9',
                                     'Value': '18',
                                     'Weight': '3',
                                     'archive_revision_number': '7',
                                     'archived_attachment_count': '1',
                                     'record_keeper_group': 'Team C'}},
                         {'source_row': 9,
                          'values': {'ProductName': '10',
                                     'Value': '22',
                                     'Weight': '4',
                                     'archive_revision_number': '5',
                                     'archived_attachment_count': '2',
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

def solve_problem():
    import pandas as pd
    capacity_df = CSVQA_FRAMES['file_0_view_0']
    products_df = CSVQA_FRAMES['file_1_view_0']
    S = list(capacity_df['SectionID'])
    P = list(products_df['ProductName'])
    C_s = {}
    for (idx, row) in capacity_df.iterrows():
        section_id = row['SectionID']
        try:
            cap = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for SectionID {section_id}: {row['Capacity']}")
        C_s[section_id] = cap
    v_p = {}
    w_p = {}
    for (idx, row) in products_df.iterrows():
        product = row['ProductName']
        try:
            val = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {product}: {row['Value']}")
        try:
            wt = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {product}: {row['Weight']}")
        v_p[product] = val
        w_p[product] = wt
    for s in S:
        if s not in C_s:
            raise ValueError(f'Missing capacity for section {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('Supermarket_Section_Stocking')
    quantity_keys = [(s, p) for s in S for p in P]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * quantity_vars[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    for s in S:
        m.addConstr(gp.quicksum((w_p[p] * quantity_vars[s, p] for p in P)) <= C_s[s])
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')