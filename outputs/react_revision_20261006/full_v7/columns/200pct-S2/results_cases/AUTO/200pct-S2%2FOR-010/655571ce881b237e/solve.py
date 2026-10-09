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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    sections_df = CSVQA_FRAMES['file_0_view_0']
    products_df = CSVQA_FRAMES['file_1_view_0']
    I = list(sections_df['SectionID'])
    J = list(products_df['ProductName'])
    c_i = {}
    for (idx, row) in sections_df.iterrows():
        section_id = row['SectionID']
        try:
            c_i[section_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for SectionID {section_id}: {row['Capacity']}")
    v_j = {}
    w_j = {}
    for (idx, row) in products_df.iterrows():
        product_name = row['ProductName']
        try:
            v_j[product_name] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {product_name}: {row['Value']}")
        try:
            w_j[product_name] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {product_name}: {row['Weight']}")
    if set(I) != set(c_i.keys()):
        raise ValueError('SectionID mismatch between index set and capacity dictionary')
    if set(J) != set(v_j.keys()) or set(J) != set(w_j.keys()):
        raise ValueError('ProductName mismatch between index set and value/weight dictionaries')
    m = gp.Model('Supermarket_Section_Stocking')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')