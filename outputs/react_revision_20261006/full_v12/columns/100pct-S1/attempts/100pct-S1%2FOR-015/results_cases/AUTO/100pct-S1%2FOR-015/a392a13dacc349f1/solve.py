CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of BigMart Sales, the store needs to allocate various types of products into different '
          'display shelves. Specifically, the store has several shelves, each with a capacity limit provided in '
          '‚Äúcapacity.csv.‚Äù The predefined value and weight of each product can be found in ‚Äúproducts.csv.‚Äù The '
          'objective is to determine the optimal number of units of each product to place on each shelf to maximize '
          'the total value of the products across all shelves, while ensuring that the total weight of the products on '
          'each shelf does not exceed its capacity. The decision variables x_ij represent the number of units of '
          'product j to be placed on shelf i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['resource_id', 'resource_capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'resource_capacity': '500', 'resource_id': '1'}},
                         {'source_row': 1, 'values': {'resource_capacity': '700', 'resource_id': '2'}},
                         {'source_row': 2, 'values': {'resource_capacity': '600', 'resource_id': '3'}},
                         {'source_row': 3, 'values': {'resource_capacity': '800', 'resource_id': '4'}},
                         {'source_row': 4, 'values': {'resource_capacity': '550', 'resource_id': '5'}},
                         {'source_row': 5, 'values': {'resource_capacity': '900', 'resource_id': '6'}},
                         {'source_row': 6, 'values': {'resource_capacity': '650', 'resource_id': '7'}},
                         {'source_row': 7, 'values': {'resource_capacity': '750', 'resource_id': '8'}},
                         {'source_row': 8, 'values': {'resource_capacity': '820', 'resource_id': '9'}},
                         {'source_row': 9, 'values': {'resource_capacity': '570', 'resource_id': '10'}}],
             'returned_rows': 10,
             'role': 'shelf capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['item_name', 'item_value', 'resource_requirement'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'item_name': '1', 'item_value': '50', 'resource_requirement': '10'}},
                         {'source_row': 1,
                          'values': {'item_name': '2', 'item_value': '70', 'resource_requirement': '20'}},
                         {'source_row': 2,
                          'values': {'item_name': '3', 'item_value': '30', 'resource_requirement': '5'}},
                         {'source_row': 3,
                          'values': {'item_name': '4', 'item_value': '60', 'resource_requirement': '15'}},
                         {'source_row': 4,
                          'values': {'item_name': '5', 'item_value': '80', 'resource_requirement': '25'}},
                         {'source_row': 5,
                          'values': {'item_name': '6', 'item_value': '90', 'resource_requirement': '30'}},
                         {'source_row': 6,
                          'values': {'item_name': '7', 'item_value': '40', 'resource_requirement': '12'}},
                         {'source_row': 7,
                          'values': {'item_name': '8', 'item_value': '100', 'resource_requirement': '35'}},
                         {'source_row': 8,
                          'values': {'item_name': '9', 'item_value': '55', 'resource_requirement': '10'}},
                         {'source_row': 9,
                          'values': {'item_name': '10', 'item_value': '75', 'resource_requirement': '20'}},
                         {'source_row': 10,
                          'values': {'item_name': '11', 'item_value': '65', 'resource_requirement': '18'}},
                         {'source_row': 11,
                          'values': {'item_name': '12', 'item_value': '95', 'resource_requirement': '28'}},
                         {'source_row': 12,
                          'values': {'item_name': '13', 'item_value': '45', 'resource_requirement': '8'}},
                         {'source_row': 13,
                          'values': {'item_name': '14', 'item_value': '85', 'resource_requirement': '22'}},
                         {'source_row': 14,
                          'values': {'item_name': '15', 'item_value': '70', 'resource_requirement': '25'}},
                         {'source_row': 15,
                          'values': {'item_name': '16', 'item_value': '110', 'resource_requirement': '40'}},
                         {'source_row': 16,
                          'values': {'item_name': '17', 'item_value': '50', 'resource_requirement': '14'}},
                         {'source_row': 17,
                          'values': {'item_name': '18', 'item_value': '60', 'resource_requirement': '16'}},
                         {'source_row': 18,
                          'values': {'item_name': '19', 'item_value': '120', 'resource_requirement': '50'}},
                         {'source_row': 19,
                          'values': {'item_name': '20', 'item_value': '100', 'resource_requirement': '30'}}],
             'returned_rows': 20,
             'role': 'product parameters',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    shelves_frame = CSVQA_FRAMES['file_0_view_0']
    products_frame = CSVQA_FRAMES['file_1_view_0']
    S = []
    C_s = {}
    for (_, row) in shelves_frame.iterrows():
        s = row['resource_id']
        S.append(s)
        C_s[s] = float(row['resource_capacity'])
    P = []
    v_p = {}
    a_p = {}
    for (_, row) in products_frame.iterrows():
        p = row['item_name']
        P.append(p)
        v_p[p] = float(row['item_value'])
        a_p[p] = float(row['resource_requirement'])
    m = gp.Model('BigMart_Shelf_Allocation')
    quantity_vars = m.addVars(S, P, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * quantity_vars[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((a_p[p] * quantity_vars[s, p] for p in P)) <= C_s[s] for s in S), name='')
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