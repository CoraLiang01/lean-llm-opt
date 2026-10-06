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
 'tables': [{'columns': ['section_light_inspections_last_year',
                         'section_signage_updates_last_year',
                         'SectionID',
                         'section_cleaning_minutes_last_month',
                         'aisle_signage_count',
                         'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '100',
                                     'SectionID': '1',
                                     'aisle_signage_count': '6',
                                     'section_cleaning_minutes_last_month': '180',
                                     'section_light_inspections_last_year': '3',
                                     'section_signage_updates_last_year': '2'}},
                         {'source_row': 1,
                          'values': {'Capacity': '150',
                                     'SectionID': '2',
                                     'aisle_signage_count': '4',
                                     'section_cleaning_minutes_last_month': '300',
                                     'section_light_inspections_last_year': '4',
                                     'section_signage_updates_last_year': '4'}},
                         {'source_row': 2,
                          'values': {'Capacity': '120',
                                     'SectionID': '3',
                                     'aisle_signage_count': '5',
                                     'section_cleaning_minutes_last_month': '300',
                                     'section_light_inspections_last_year': '6',
                                     'section_signage_updates_last_year': '6'}},
                         {'source_row': 3,
                          'values': {'Capacity': '130',
                                     'SectionID': '4',
                                     'aisle_signage_count': '5',
                                     'section_cleaning_minutes_last_month': '180',
                                     'section_light_inspections_last_year': '8',
                                     'section_signage_updates_last_year': '4'}},
                         {'source_row': 4,
                          'values': {'Capacity': '90',
                                     'SectionID': '5',
                                     'aisle_signage_count': '4',
                                     'section_cleaning_minutes_last_month': '300',
                                     'section_light_inspections_last_year': '3',
                                     'section_signage_updates_last_year': '6'}},
                         {'source_row': 5,
                          'values': {'Capacity': '110',
                                     'SectionID': '6',
                                     'aisle_signage_count': '5',
                                     'section_cleaning_minutes_last_month': '300',
                                     'section_light_inspections_last_year': '6',
                                     'section_signage_updates_last_year': '3'}},
                         {'source_row': 6,
                          'values': {'Capacity': '160',
                                     'SectionID': '7',
                                     'aisle_signage_count': '4',
                                     'section_cleaning_minutes_last_month': '360',
                                     'section_light_inspections_last_year': '8',
                                     'section_signage_updates_last_year': '3'}},
                         {'source_row': 7,
                          'values': {'Capacity': '140',
                                     'SectionID': '8',
                                     'aisle_signage_count': '4',
                                     'section_cleaning_minutes_last_month': '300',
                                     'section_light_inspections_last_year': '6',
                                     'section_signage_updates_last_year': '6'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['merchandising_theme',
                         'packaging_label_review_count',
                         'marketing_campaign_format',
                         'ProductName',
                         'Value',
                         'product_catalog_page_views',
                         'supplier_contact_channel',
                         'Weight',
                         'supplier_catalog_revision_count'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'ProductName': '1',
                                     'Value': '10',
                                     'Weight': '2',
                                     'marketing_campaign_format': 'Newsletter',
                                     'merchandising_theme': 'Featured',
                                     'packaging_label_review_count': '7',
                                     'product_catalog_page_views': '1040',
                                     'supplier_catalog_revision_count': '4',
                                     'supplier_contact_channel': 'Phone'}},
                         {'source_row': 1,
                          'values': {'ProductName': '2',
                                     'Value': '15',
                                     'Weight': '3',
                                     'marketing_campaign_format': 'Brochure',
                                     'merchandising_theme': 'Seasonal',
                                     'packaging_label_review_count': '5',
                                     'product_catalog_page_views': '340',
                                     'supplier_catalog_revision_count': '6',
                                     'supplier_contact_channel': 'Portal'}},
                         {'source_row': 2,
                          'values': {'ProductName': '3',
                                     'Value': '8',
                                     'Weight': '1',
                                     'marketing_campaign_format': 'Brochure',
                                     'merchandising_theme': 'Everyday',
                                     'packaging_label_review_count': '7',
                                     'product_catalog_page_views': '180',
                                     'supplier_catalog_revision_count': '2',
                                     'supplier_contact_channel': 'Portal'}},
                         {'source_row': 3,
                          'values': {'ProductName': '4',
                                     'Value': '12',
                                     'Weight': '2',
                                     'marketing_campaign_format': 'Brochure',
                                     'merchandising_theme': 'Seasonal',
                                     'packaging_label_review_count': '5',
                                     'product_catalog_page_views': '1380',
                                     'supplier_catalog_revision_count': '4',
                                     'supplier_contact_channel': 'Phone'}},
                         {'source_row': 4,
                          'values': {'ProductName': '5',
                                     'Value': '20',
                                     'Weight': '4',
                                     'marketing_campaign_format': 'Brochure',
                                     'merchandising_theme': 'Seasonal',
                                     'packaging_label_review_count': '5',
                                     'product_catalog_page_views': '1380',
                                     'supplier_catalog_revision_count': '6',
                                     'supplier_contact_channel': 'Portal'}},
                         {'source_row': 5,
                          'values': {'ProductName': '6',
                                     'Value': '25',
                                     'Weight': '5',
                                     'marketing_campaign_format': 'Newsletter',
                                     'merchandising_theme': 'Everyday',
                                     'packaging_label_review_count': '5',
                                     'product_catalog_page_views': '1040',
                                     'supplier_catalog_revision_count': '1',
                                     'supplier_contact_channel': 'Phone'}},
                         {'source_row': 6,
                          'values': {'ProductName': '7',
                                     'Value': '5',
                                     'Weight': '1',
                                     'marketing_campaign_format': 'Web feature',
                                     'merchandising_theme': 'Seasonal',
                                     'packaging_label_review_count': '3',
                                     'product_catalog_page_views': '560',
                                     'supplier_catalog_revision_count': '1',
                                     'supplier_contact_channel': 'Phone'}},
                         {'source_row': 7,
                          'values': {'ProductName': '8',
                                     'Value': '30',
                                     'Weight': '6',
                                     'marketing_campaign_format': 'Web feature',
                                     'merchandising_theme': 'Seasonal',
                                     'packaging_label_review_count': '2',
                                     'product_catalog_page_views': '790',
                                     'supplier_catalog_revision_count': '3',
                                     'supplier_contact_channel': 'Email'}},
                         {'source_row': 8,
                          'values': {'ProductName': '9',
                                     'Value': '18',
                                     'Weight': '3',
                                     'marketing_campaign_format': 'Newsletter',
                                     'merchandising_theme': 'Seasonal',
                                     'packaging_label_review_count': '3',
                                     'product_catalog_page_views': '340',
                                     'supplier_catalog_revision_count': '3',
                                     'supplier_contact_channel': 'Portal'}},
                         {'source_row': 9,
                          'values': {'ProductName': '10',
                                     'Value': '22',
                                     'Weight': '4',
                                     'marketing_campaign_format': 'Brochure',
                                     'merchandising_theme': 'Featured',
                                     'packaging_label_review_count': '5',
                                     'product_catalog_page_views': '340',
                                     'supplier_catalog_revision_count': '2',
                                     'supplier_contact_channel': 'Phone'}}],
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
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    section_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            section_table = t
            break
    if section_table is None:
        raise RuntimeError('Section table not found')
    S = []
    C = {}
    for rec in section_table['records']:
        sid = rec['values']['SectionID']
        cap = rec['values']['Capacity']
        S.append(sid)
        C[sid] = float(cap)
    product_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_1_view_0':
            product_table = t
            break
    if product_table is None:
        raise RuntimeError('Product table not found')
    P = []
    v = {}
    w = {}
    for rec in product_table['records']:
        pname = rec['values']['ProductName']
        val = rec['values']['Value']
        wei = rec['values']['Weight']
        P.append(pname)
        v[pname] = float(val)
        w[pname] = float(wei)
    for sid in S:
        if sid not in C:
            raise ValueError(f'Missing capacity for section {sid}')
    for pname in P:
        if pname not in v or pname not in w:
            raise ValueError(f'Missing value or weight for product {pname}')
    m = gp.Model('Supermarket_Section_Stocking')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(S, P, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[p] * x[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[p] * x[s, p] for p in P)) <= C[s] for s in S), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()