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
 'route': 'RA',
 'tables': [{'columns': ['Equipment / Cost',
                         'Product I',
                         'Product II',
                         'Product III',
                         'Available Equipment Operating Time',
                         'Equipment Cost at Full Load (yuan)'],
             'file_index': 0,
             'file_name': '43.csv',
             'filters': {'conditions': [{'column': 'Equipment / Cost',
                                         'dtype': 'string',
                                         'evidence': 'A1, A2, B1, B2, B3',
                                         'inclusive': 'both',
                                         'operator': 'in',
                                         'value': ['A1', 'A2', 'B1', 'B2', 'B3']}],
                         'logic': 'and'},
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
                                     'Product III': ''}}],
             'returned_rows': 5,
             'role': 'equipment-product processing and cost matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA['tables'][0]['records']
    products = ['Product I', 'Product II', 'Product III']
    product_ids = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
    E_A = ['A1', 'A2']
    E_B = ['B1', 'B2', 'B3']
    equipment = E_A + E_B
    E_A_p = {'I': ['A1', 'A2'], 'II': ['A1', 'A2'], 'III': ['A2']}
    E_B_p = {'I': ['B1', 'B2', 'B3'], 'II': ['B1'], 'III': ['B2']}
    t_pe = {}
    T_e = {}
    C_e = {}
    for rec in data:
        e = rec['values']['Equipment / Cost']
        T_e[e] = float(rec['values']['Available Equipment Operating Time'])
        C_e[e] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
        for idx, p in enumerate(products):
            p_id = product_ids[p]
            val = rec['values'][p]
            if val != '' and val is not None:
                t_pe[p_id, e] = float(val)
    raise ValueError('Raw material cost and selling price per product are not present in the provided data (43.csv). Cannot proceed with optimization.')
m = solve_problem()