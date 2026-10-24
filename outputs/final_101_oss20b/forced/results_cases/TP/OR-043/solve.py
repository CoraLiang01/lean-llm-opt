CSVQA_DATA = {'query': 'A pharmacy chain needs to replenish its drug inventory. The ‘products.csv’ file provides a table of benefits associated with each drug product. There are overall stock capacity limits for pharmacy chains detailed in the ‘capacity.csv’ file.Our objective is to decide which drugs to order each day and in what quantities to maximise the overall benefit while adhering to the overall stock capacity. The decision variable x_i represents the number of units of the ith drug to be ordered each day.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '520'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'NSAIDs', 'Value': '250', 'Weight': '913'}}, {'data_row_number': 1, 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '178', 'Weight': '754'}}, {'data_row_number': 2, 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '313', 'Weight': '428'}}, {'data_row_number': 3, 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '711'}}, {'data_row_number': 4, 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'data_row_number': 5, 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'data_row_number': 6, 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'data_row_number': 7, 'values': {'ProductName': 'Antipsychotics', 'Value': '934', 'Weight': '291'}}, {'data_row_number': 8, 'values': {'ProductName': 'Antihistamines', 'Value': '114', 'Weight': '302'}}, {'data_row_number': 9, 'values': {'ProductName': 'Corticosteroids', 'Value': '1357', 'Weight': '50'}}, {'data_row_number': 10, 'values': {'ProductName': 'Beta Blockers', 'Value': '156', 'Weight': '250'}}, {'data_row_number': 11, 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '1780', 'Weight': '178'}}, {'data_row_number': 12, 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'data_row_number': 13, 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'data_row_number': 14, 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'data_row_number': 15, 'values': {'ProductName': 'Statins', 'Value': '320', 'Weight': '97'}}, {'data_row_number': 16, 'values': {'ProductName': 'Insulin', 'Value': '1357', 'Weight': '470'}}, {'data_row_number': 17, 'values': {'ProductName': 'Anticoagulants', 'Value': '1357', 'Weight': '341'}}, {'data_row_number': 18, 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '405', 'Weight': '121'}}, {'data_row_number': 19, 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    capacity_table = None
    products_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            capacity_table = tbl
        elif tbl.get('table_id') == 'file_1_view_0':
            products_table = tbl
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found in CSVQA_DATA.")
    if products_table is None:
        raise ValueError("Products table 'file_1_view_0' not found in CSVQA_DATA.")
    cap_records = capacity_table.get('records', [])
    if not cap_records:
        raise ValueError('No records found in capacity table.')
    cap_value_str = cap_records[0].get('values', {}).get('Capacity')
    if cap_value_str is None:
        raise ValueError('Capacity value missing.')
    try:
        C = float(cap_value_str)
    except ValueError:
        raise ValueError(f'Invalid capacity value: {cap_value_str}')
    product_records = products_table.get('records', [])
    if not product_records:
        raise ValueError('No product records found.')
    product_names = []
    c_dict = {}
    w_dict = {}
    for rec in product_records:
        vals = rec.get('values', {})
        pname = vals.get('ProductName')
        if pname is None:
            raise ValueError('ProductName missing in a record.')
        if pname in product_names:
            raise ValueError(f'Duplicate product name detected: {pname}')
        product_names.append(pname)
        val_str = vals.get('Value')
        if val_str is None:
            raise ValueError(f'Value missing for product {pname}.')
        try:
            c_val = float(val_str)
        except ValueError:
            raise ValueError(f'Invalid Value for product {pname}: {val_str}')
        c_dict[pname] = c_val
        w_str = vals.get('Weight')
        if w_str is None:
            raise ValueError(f'Weight missing for product {pname}.')
        try:
            w_val = float(w_str)
        except ValueError:
            raise ValueError(f'Invalid Weight for product {pname}: {w_str}')
        w_dict[pname] = w_val
    m = gurobipy.Model('PharmacyInventory')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((c_dict[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w_dict[p] * x[p] for p in product_names)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
