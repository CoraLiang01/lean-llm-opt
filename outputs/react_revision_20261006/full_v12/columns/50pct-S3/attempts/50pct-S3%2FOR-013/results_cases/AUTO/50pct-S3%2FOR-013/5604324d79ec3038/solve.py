CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Amazon needs to allocate different types of air conditioners into different warehouse storage areas. '
          'Specifically, Amazon has several storage areas, each with a capacity limit provided in ‚Äúcapacity.csv.‚Äù '
          'The predefined value and size of each air conditioner type can be found in ‚Äúproducts.csv.‚Äù The '
          'objective is to determine the optimal number of units of each air conditioner type to place in each storage '
          'area to maximize the total value of the air conditioners across all areas, while ensuring that the total '
          'size of the units in each area does not exceed its capacity. The decision variablesx_ijrepresent the number '
          'of units of air conditioner type j to be placed in storage area i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['StorageID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 15,
             'records': [{'source_row': 0, 'values': {'Capacity': '1083', 'StorageID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '1840', 'StorageID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '770', 'StorageID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '1299', 'StorageID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '1259', 'StorageID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '543', 'StorageID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '1831', 'StorageID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '855', 'StorageID': '8'}},
                         {'source_row': 8, 'values': {'Capacity': '619', 'StorageID': '9'}},
                         {'source_row': 9, 'values': {'Capacity': '637', 'StorageID': '10'}},
                         {'source_row': 10, 'values': {'Capacity': '935', 'StorageID': '11'}},
                         {'source_row': 11, 'values': {'Capacity': '626', 'StorageID': '12'}},
                         {'source_row': 12, 'values': {'Capacity': '1457', 'StorageID': '13'}},
                         {'source_row': 13, 'values': {'Capacity': '1198', 'StorageID': '14'}},
                         {'source_row': 14, 'values': {'Capacity': '837', 'StorageID': '15'}}],
             'returned_rows': 15,
             'role': 'storage area capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Window Unit', 'Value': '4811', 'Weight': '114'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Portable Unit', 'Value': '1130', 'Weight': '200'}},
                         {'source_row': 2, 'values': {'ProductName': 'Split System', 'Value': '1611', 'Weight': '106'}},
                         {'source_row': 3,
                          'values': {'ProductName': 'Ductless System', 'Value': '3368', 'Weight': '256'}},
                         {'source_row': 4, 'values': {'ProductName': 'Central AC', 'Value': '2135', 'Weight': '268'}},
                         {'source_row': 5, 'values': {'ProductName': 'Hybrid AC', 'Value': '1046', 'Weight': '185'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Geothermal AC', 'Value': '4030', 'Weight': '299'}},
                         {'source_row': 7, 'values': {'ProductName': 'Smart AC', 'Value': '3761', 'Weight': '131'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Evaporative Cooler', 'Value': '3523', 'Weight': '139'}},
                         {'source_row': 9,
                          'values': {'ProductName': 'Package Unit', 'Value': '1701', 'Weight': '105'}}],
             'returned_rows': 10,
             'role': 'air conditioner product parameters',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_frame = CSVQA_FRAMES['file_0_view_0']
    products_frame = CSVQA_FRAMES['file_1_view_0']
    I = []
    c = {}
    for (_, row) in capacity_frame.iterrows():
        storage_id = row['StorageID']
        I.append(storage_id)
        c[storage_id] = float(row['Capacity'])
    J = []
    v = {}
    w = {}
    for (_, row) in products_frame.iterrows():
        product_name = row['ProductName']
        J.append(product_name)
        v[product_name] = float(row['Value'])
        w[product_name] = float(row['Weight'])
    m = gp.Model('Amazon_AC_Storage_Allocation')
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * quantity_vars[i, j] for j in J)) <= c[i] for i in I), name='')
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