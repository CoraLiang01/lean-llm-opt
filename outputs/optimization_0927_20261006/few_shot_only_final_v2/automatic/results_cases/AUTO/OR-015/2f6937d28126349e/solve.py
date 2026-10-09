CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 7,
             'returned_rows': 7,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Aalopuri',
                                     'Revenue': '20',
                                     'Demand': '1483',
                                     'Initial Inventory': '10440.0'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Cold coffee',
                                     'Revenue': '40',
                                     'Demand': '1918',
                                     'Initial Inventory': '13610.0'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Frankie',
                                     'Revenue': '50',
                                     'Demand': '1623',
                                     'Initial Inventory': '11500.0'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Panipuri',
                                     'Revenue': '20',
                                     'Demand': '1720',
                                     'Initial Inventory': '12260.0'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Sandwich',
                                     'Revenue': '60',
                                     'Demand': '1558',
                                     'Initial Inventory': '10970.0'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Sugarcane juice',
                                     'Revenue': '25',
                                     'Demand': '1791',
                                     'Initial Inventory': '12780.0'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Vadapav',
                                     'Revenue': '20',
                                     'Demand': '1426',
                                     'Initial Inventory': '10060.0'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import gurobipy as gp
from gurobipy import GRB
import re
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError('Table file_0_view_0 not found in CSVQA_DATA.')
records = table['records']
if not records:
    raise ValueError('No records found in file_0_view_0.')
aalop_indices = []
for (idx, rec) in enumerate(records):
    pname = rec['values'].get('Product Name', '')
    if re.search('\\bAalop\\b', pname, re.IGNORECASE):
        aalop_indices.append(idx)
if not aalop_indices:
    raise ValueError("No products classified under 'Aalop' found.")
I = []
A = {}
d = {}
Iinv = {}
for idx in aalop_indices:
    rec = records[idx]
    pname = rec['values']['Product Name']
    I.append(pname)
    try:
        A[pname] = float(rec['values']['Revenue'])
    except Exception:
        raise ValueError(f"Invalid Revenue for product '{pname}'.")
    try:
        d[pname] = int(float(rec['values']['Demand']))
    except Exception:
        raise ValueError(f"Invalid Demand for product '{pname}'.")
    try:
        Iinv[pname] = int(float(rec['values']['Initial Inventory']))
    except Exception:
        raise ValueError(f"Invalid Initial Inventory for product '{pname}'.")
for pname in I:
    if pname not in A or pname not in d or pname not in Iinv:
        raise ValueError(f"Missing data for product '{pname}'.")

def build_and_solve(I, A, d, Iinv):
    m = gp.Model('Aalop_Product_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= Iinv[i] for i in I), name='')
    m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in x_vars.values():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve(I, A, d, Iinv)