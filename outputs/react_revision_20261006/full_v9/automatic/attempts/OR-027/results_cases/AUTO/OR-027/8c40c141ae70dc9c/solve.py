CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 23,
             'records': [{'source_row': 0,
                          'values': {'Demand': '715183',
                                     'Initial Inventory': '5346490.0',
                                     'Revenue': '165.2',
                                     'Sub Category': 'Atta & Flour'}},
                         {'source_row': 1,
                          'values': {'Demand': '924398',
                                     'Initial Inventory': '6840830.0',
                                     'Revenue': '181.93',
                                     'Sub Category': 'Biscuits'}},
                         {'source_row': 2,
                          'values': {'Demand': '1006220',
                                     'Initial Inventory': '7425860.0',
                                     'Revenue': '189.99',
                                     'Sub Category': 'Breads & Buns'}},
                         {'source_row': 3,
                          'values': {'Demand': '924932',
                                     'Initial Inventory': '6856120.0',
                                     'Revenue': '484.65',
                                     'Sub Category': 'Cakes'}},
                         {'source_row': 4,
                          'values': {'Demand': '702263',
                                     'Initial Inventory': '5204970.0',
                                     'Revenue': '207.75',
                                     'Sub Category': 'Chicken'}},
                         {'source_row': 5,
                          'values': {'Demand': '994869',
                                     'Initial Inventory': '7338980.0',
                                     'Revenue': '437.69',
                                     'Sub Category': 'Chocolates'}},
                         {'source_row': 6,
                          'values': {'Demand': '1031871',
                                     'Initial Inventory': '7682130.0',
                                     'Revenue': '315.21',
                                     'Sub Category': 'Cookies'}},
                         {'source_row': 7,
                          'values': {'Demand': '714036',
                                     'Initial Inventory': '5233710.0',
                                     'Revenue': '47.4',
                                     'Sub Category': 'Dals & Pulses'}},
                         {'source_row': 8,
                          'values': {'Demand': '900971',
                                     'Initial Inventory': '6680860.0',
                                     'Revenue': '100.8',
                                     'Sub Category': 'Edible Oil & Ghee'}},
                         {'source_row': 9,
                          'values': {'Demand': '774463',
                                     'Initial Inventory': '5751560.0',
                                     'Revenue': '308.44',
                                     'Sub Category': 'Eggs'}},
                         {'source_row': 10,
                          'values': {'Demand': '756970',
                                     'Initial Inventory': '5605480.0',
                                     'Revenue': '271.17',
                                     'Sub Category': 'Fish'}},
                         {'source_row': 11,
                          'values': {'Demand': '738560',
                                     'Initial Inventory': '5512120.0',
                                     'Revenue': '147.76',
                                     'Sub Category': 'Fresh Fruits'}},
                         {'source_row': 12,
                          'values': {'Demand': '709643',
                                     'Initial Inventory': '5258420.0',
                                     'Revenue': '89.6',
                                     'Sub Category': 'Fresh Vegetables'}},
                         {'source_row': 13,
                          'values': {'Demand': '1419411',
                                     'Initial Inventory': '10514390.0',
                                     'Revenue': '149.8',
                                     'Sub Category': 'Health Drinks'}},
                         {'source_row': 14,
                          'values': {'Demand': '937261',
                                     'Initial Inventory': '6974800.0',
                                     'Revenue': '401.28',
                                     'Sub Category': 'Masalas'}},
                         {'source_row': 15,
                          'values': {'Demand': '818578',
                                     'Initial Inventory': '6112000.0',
                                     'Revenue': '386.54',
                                     'Sub Category': 'Mutton'}},
                         {'source_row': 16,
                          'values': {'Demand': '996569',
                                     'Initial Inventory': '7354350.0',
                                     'Revenue': '420.0',
                                     'Sub Category': 'Noodles'}},
                         {'source_row': 17,
                          'values': {'Demand': '678906',
                                     'Initial Inventory': '5034020.0',
                                     'Revenue': '60.8',
                                     'Sub Category': 'Organic Fruits'}},
                         {'source_row': 18,
                          'values': {'Demand': '749927',
                                     'Initial Inventory': '5589290.0',
                                     'Revenue': '918.45',
                                     'Sub Category': 'Organic Staples'}},
                         {'source_row': 19,
                          'values': {'Demand': '699808',
                                     'Initial Inventory': '5202710.0',
                                     'Revenue': '77.52',
                                     'Sub Category': 'Organic Vegetables'}},
                         {'source_row': 20,
                          'values': {'Demand': '674626',
                                     'Initial Inventory': '4983230.0',
                                     'Revenue': '615.65',
                                     'Sub Category': 'Rice'}},
                         {'source_row': 21,
                          'values': {'Demand': '1389437',
                                     'Initial Inventory': '10338740.0',
                                     'Revenue': '159.16',
                                     'Sub Category': 'Soft Drinks'}},
                         {'source_row': 22,
                          'values': {'Demand': '905834',
                                     'Initial Inventory': '6728760.0',
                                     'Revenue': '227.7',
                                     'Sub Category': 'Spices'}}],
             'returned_rows': 23,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': 'expected a string or tuple, not list',
                'planner_errors': ['expected a string or tuple, not list'],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    organ_indices = []
    revenue = {}
    demand = {}
    inventory = {}
    for (source_row, row) in frame.iterrows():
        subcat = row['Sub Category']
        if isinstance(subcat, str) and 'organ' in subcat.casefold():
            idx = source_row
            organ_indices.append(idx)
            try:
                revenue[idx] = float(row['Revenue'])
            except Exception:
                raise ValueError(f'Missing or invalid Revenue for row {idx}')
            try:
                demand[idx] = int(float(row['Demand']))
            except Exception:
                raise ValueError(f'Missing or invalid Demand for row {idx}')
            try:
                inventory[idx] = int(float(row['Initial Inventory']))
            except Exception:
                raise ValueError(f'Missing or invalid Initial Inventory for row {idx}')
    if not organ_indices:
        raise ValueError("No products with 'Organ' in 'Sub Category' found.")
    for idx in organ_indices:
        if idx not in revenue or idx not in demand or idx not in inventory:
            raise ValueError(f'Missing data for product at row {idx}')
    m = gp.Model('Organ_Product_Revenue_Maximization')
    x_vars = m.addVars(organ_indices, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in organ_indices)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= demand[i] for i in organ_indices), name='')
    m.addConstrs((x_vars[i] <= inventory[i] for i in organ_indices), name='')
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