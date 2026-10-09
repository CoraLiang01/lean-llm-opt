CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 12,
             'returned_rows': 12,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28',
                                     'Demand': '3066513',
                                     'Initial Inventory': '22749210'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Beverages_47.45',
                                     'Revenue': '47.45',
                                     'Demand': '2961484',
                                     'Initial Inventory': '22049510'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Cereal_205.7',
                                     'Revenue': '205.7',
                                     'Demand': '2621950',
                                     'Initial Inventory': '19459680'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Clothes_109.28',
                                     'Revenue': '109.28',
                                     'Demand': '2660974',
                                     'Initial Inventory': '19754410'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Cosmetics_437.2',
                                     'Revenue': '437.2',
                                     'Demand': '2896197',
                                     'Initial Inventory': '21366410'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Fruits_9.33',
                                     'Revenue': '9.33',
                                     'Demand': '3169426',
                                     'Initial Inventory': '23410830'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Household_668.27',
                                     'Revenue': '668.27',
                                     'Demand': '2846953',
                                     'Initial Inventory': '20986130'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'Meat_421.89',
                                     'Revenue': '421.89',
                                     'Demand': '2546972',
                                     'Initial Inventory': '19011970'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'Office Supplies_651.21',
                                     'Revenue': '651.21',
                                     'Demand': '2855686',
                                     'Initial Inventory': '21062780'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'Personal Care_81.73',
                                     'Revenue': '81.73',
                                     'Demand': '2855360',
                                     'Initial Inventory': '21265920'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'Snacks_152.58',
                                     'Revenue': '152.58',
                                     'Demand': '2592261',
                                     'Initial Inventory': '19155280'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'Vegetables_154.06',
                                     'Revenue': '154.06',
                                     'Demand': '2826603',
                                     'Initial Inventory': '20867490'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
table_id = 'file_0_view_0'
tables = [t for t in CSVQA_DATA['tables'] if t['table_id'] == table_id]
if not tables:
    raise ValueError(f'Table {table_id} not found in CSVQA_DATA.')
records = tables[0]['records']
baby_regex = re.compile('\\bBaby\\b', re.IGNORECASE)
baby_indices = []
for (idx, rec) in enumerate(records):
    pname = rec['values'].get('Product Name', '')
    if baby_regex.search(str(pname)):
        baby_indices.append(idx)
if not baby_indices:
    raise ValueError("No 'Baby' products found in table file_0_view_0.")
B = []
A = {}
d = {}
I = {}
for idx in baby_indices:
    rec = records[idx]
    pname = rec['values'].get('Product Name', '')
    B.append(pname)
    try:
        revenue = rec['values'].get('Revenue', '')
        demand = rec['values'].get('Demand', '')
        inventory = rec['values'].get('Initial Inventory', '')
        if revenue == '' or demand == '' or inventory == '':
            raise ValueError
        A[pname] = float(revenue)
        d[pname] = int(float(demand))
        I[pname] = int(float(inventory))
    except Exception:
        raise ValueError(f"Missing or invalid numeric data for product '{pname}'.")
if not set(A.keys()) == set(B) == set(d.keys()) == set(I.keys()):
    raise ValueError('Parameter keys do not match index set B.')
m = gp.Model('Baby_Product_Revenue_Max')
x_vars = m.addVars(B, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in B)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I[i] for i in B), name='')
m.addConstrs((x_vars[i] <= d[i] for i in B), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')