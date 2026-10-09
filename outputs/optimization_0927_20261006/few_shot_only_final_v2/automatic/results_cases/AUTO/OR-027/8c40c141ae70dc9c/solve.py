CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'role': 'file_0',
             'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 23,
             'returned_rows': 23,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Sub Category': 'Atta & Flour',
                                     'Revenue': '165.2',
                                     'Demand': '715183',
                                     'Initial Inventory': '5346490.0'}},
                         {'source_row': 1,
                          'values': {'Sub Category': 'Biscuits',
                                     'Revenue': '181.93',
                                     'Demand': '924398',
                                     'Initial Inventory': '6840830.0'}},
                         {'source_row': 2,
                          'values': {'Sub Category': 'Breads & Buns',
                                     'Revenue': '189.99',
                                     'Demand': '1006220',
                                     'Initial Inventory': '7425860.0'}},
                         {'source_row': 3,
                          'values': {'Sub Category': 'Cakes',
                                     'Revenue': '484.65',
                                     'Demand': '924932',
                                     'Initial Inventory': '6856120.0'}},
                         {'source_row': 4,
                          'values': {'Sub Category': 'Chicken',
                                     'Revenue': '207.75',
                                     'Demand': '702263',
                                     'Initial Inventory': '5204970.0'}},
                         {'source_row': 5,
                          'values': {'Sub Category': 'Chocolates',
                                     'Revenue': '437.69',
                                     'Demand': '994869',
                                     'Initial Inventory': '7338980.0'}},
                         {'source_row': 6,
                          'values': {'Sub Category': 'Cookies',
                                     'Revenue': '315.21',
                                     'Demand': '1031871',
                                     'Initial Inventory': '7682130.0'}},
                         {'source_row': 7,
                          'values': {'Sub Category': 'Dals & Pulses',
                                     'Revenue': '47.4',
                                     'Demand': '714036',
                                     'Initial Inventory': '5233710.0'}},
                         {'source_row': 8,
                          'values': {'Sub Category': 'Edible Oil & Ghee',
                                     'Revenue': '100.8',
                                     'Demand': '900971',
                                     'Initial Inventory': '6680860.0'}},
                         {'source_row': 9,
                          'values': {'Sub Category': 'Eggs',
                                     'Revenue': '308.44',
                                     'Demand': '774463',
                                     'Initial Inventory': '5751560.0'}},
                         {'source_row': 10,
                          'values': {'Sub Category': 'Fish',
                                     'Revenue': '271.17',
                                     'Demand': '756970',
                                     'Initial Inventory': '5605480.0'}},
                         {'source_row': 11,
                          'values': {'Sub Category': 'Fresh Fruits',
                                     'Revenue': '147.76',
                                     'Demand': '738560',
                                     'Initial Inventory': '5512120.0'}},
                         {'source_row': 12,
                          'values': {'Sub Category': 'Fresh Vegetables',
                                     'Revenue': '89.6',
                                     'Demand': '709643',
                                     'Initial Inventory': '5258420.0'}},
                         {'source_row': 13,
                          'values': {'Sub Category': 'Health Drinks',
                                     'Revenue': '149.8',
                                     'Demand': '1419411',
                                     'Initial Inventory': '10514390.0'}},
                         {'source_row': 14,
                          'values': {'Sub Category': 'Masalas',
                                     'Revenue': '401.28',
                                     'Demand': '937261',
                                     'Initial Inventory': '6974800.0'}},
                         {'source_row': 15,
                          'values': {'Sub Category': 'Mutton',
                                     'Revenue': '386.54',
                                     'Demand': '818578',
                                     'Initial Inventory': '6112000.0'}},
                         {'source_row': 16,
                          'values': {'Sub Category': 'Noodles',
                                     'Revenue': '420.0',
                                     'Demand': '996569',
                                     'Initial Inventory': '7354350.0'}},
                         {'source_row': 17,
                          'values': {'Sub Category': 'Organic Fruits',
                                     'Revenue': '60.8',
                                     'Demand': '678906',
                                     'Initial Inventory': '5034020.0'}},
                         {'source_row': 18,
                          'values': {'Sub Category': 'Organic Staples',
                                     'Revenue': '918.45',
                                     'Demand': '749927',
                                     'Initial Inventory': '5589290.0'}},
                         {'source_row': 19,
                          'values': {'Sub Category': 'Organic Vegetables',
                                     'Revenue': '77.52',
                                     'Demand': '699808',
                                     'Initial Inventory': '5202710.0'}},
                         {'source_row': 20,
                          'values': {'Sub Category': 'Rice',
                                     'Revenue': '615.65',
                                     'Demand': '674626',
                                     'Initial Inventory': '4983230.0'}},
                         {'source_row': 21,
                          'values': {'Sub Category': 'Soft Drinks',
                                     'Revenue': '159.16',
                                     'Demand': '1389437',
                                     'Initial Inventory': '10338740.0'}},
                         {'source_row': 22,
                          'values': {'Sub Category': 'Spices',
                                     'Revenue': '227.7',
                                     'Demand': '905834',
                                     'Initial Inventory': '6728760.0'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import gurobipy as gp
from gurobipy import GRB
import re
table_id = 'file_0_view_0'
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == table_id:
        table = t
        break
if table is None:
    raise ValueError('Required table_id file_0_view_0 not found in CSVQA_DATA.')
columns = table['columns']
if not all((col in columns for col in ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'])):
    raise ValueError('Missing required columns in file_0_view_0.')
records = table['records']
organ_indices = []
for (idx, rec) in enumerate(records):
    subcat = rec['values'].get('Sub Category', '')
    if re.search('Organ', subcat, re.IGNORECASE):
        organ_indices.append(idx)
if not organ_indices:
    raise ValueError("No products with 'Organ' in 'Sub Category' found.")
A = {}
d = {}
I = {}
for idx in organ_indices:
    rec = records[idx]['values']
    key = idx
    try:
        revenue_str = rec['Revenue']
        demand_str = rec['Demand']
        inventory_str = rec['Initial Inventory']
    except KeyError as e:
        raise ValueError(f'Missing required field {e} in record {idx}.')
    try:
        A[key] = float(revenue_str)
        d[key] = float(demand_str)
        I[key] = float(inventory_str)
    except Exception as e:
        raise ValueError(f'Non-numeric value in record {idx}: {e}')
if not set(A.keys()) == set(d.keys()) == set(I.keys()) == set(organ_indices):
    raise ValueError('Mismatch in parameter keys for index set.')
m = gp.Model('Organ_Product_Revenue_Maximization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(organ_indices, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in organ_indices)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I[i] for i in organ_indices), name='')
m.addConstrs((x_vars[i] <= d[i] for i in organ_indices), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')