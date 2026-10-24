CSVQA_DATA = {'query': 'A chain pharmacy needs to restock its drug inventory. For each drug type (e.g., NSAIDs, antirheumatic drugs, acetic-acid derivatives, etc.), the benefit coefficient is listed in “products.csv.” The pharmacy faces a single overall inventory-capacity constraint, provided in “capacity.csv.”\n    The goal is to decide the daily order quantity of each drug typeso as to maximize total benefit while ensuring that the total weight of all ordered units does not exceed the overall capacity.The decision variables x_i  represent the number of units of type  i  drug to be ordered daily.The decision variables must be integers.', 'relationships': [], 'tables': [{'columns': ['Capacity'], 'file_name': 'capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '4120'}}], 'table_id': 'file_0_view_0'}, {'columns': ['ProductName', 'Value', 'Weight'], 'file_name': 'products.csv', 'records': [{'data_row_number': 0, 'values': {'ProductName': 'NSAIDs', 'Value': '585', 'Weight': '50'}}, {'data_row_number': 1, 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '557', 'Weight': '329'}}, {'data_row_number': 2, 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '963', 'Weight': '410'}}, {'data_row_number': 3, 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '452'}}, {'data_row_number': 4, 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'data_row_number': 5, 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'data_row_number': 6, 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'data_row_number': 7, 'values': {'ProductName': 'Antipsychotics', 'Value': '461', 'Weight': '291'}}, {'data_row_number': 8, 'values': {'ProductName': 'Antihistamines', 'Value': '840', 'Weight': '302'}}, {'data_row_number': 9, 'values': {'ProductName': 'Corticosteroids', 'Value': '999', 'Weight': '50'}}, {'data_row_number': 10, 'values': {'ProductName': 'Beta Blockers', 'Value': '392', 'Weight': '250'}}, {'data_row_number': 11, 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '874', 'Weight': '178'}}, {'data_row_number': 12, 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'data_row_number': 13, 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'data_row_number': 14, 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'data_row_number': 15, 'values': {'ProductName': 'Statins', 'Value': '913', 'Weight': '97'}}, {'data_row_number': 16, 'values': {'ProductName': 'Insulin', 'Value': '754', 'Weight': '470'}}, {'data_row_number': 17, 'values': {'ProductName': 'Anticoagulants', 'Value': '428', 'Weight': '341'}}, {'data_row_number': 18, 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '711', 'Weight': '121'}}, {'data_row_number': 19, 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'), None)
    if capacity_table is None:
        raise ValueError("Capacity table 'file_0_view_0' not found.")
    if not capacity_table['records']:
        raise ValueError('Capacity table has no records.')
    capacity_value = capacity_table['records'][0]['values'].get('Capacity')
    if capacity_value is None:
        raise ValueError('Capacity value missing.')
    try:
        C = int(capacity_value)
    except ValueError:
        raise ValueError(f"Capacity value '{capacity_value}' is not an integer.")
    products_table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_1_view_0'), None)
    if products_table is None:
        raise ValueError("Products table 'file_1_view_0' not found.")
    if not products_table['records']:
        raise ValueError('Products table has no records.')
    I = []
    V = {}
    W = {}
    for rec in products_table['records']:
        vals = rec['values']
        pname = vals.get('ProductName')
        if pname is None:
            raise ValueError(f"Missing ProductName in record {rec['data_row_number']}.")
        val_str = vals.get('Value')
        weight_str = vals.get('Weight')
        if val_str is None or weight_str is None:
            raise ValueError(f"Missing Value or Weight for product '{pname}'.")
        try:
            v_val = int(val_str)
            w_val = int(weight_str)
        except ValueError:
            raise ValueError(f"Non-integer Value or Weight for product '{pname}'.")
        I.append(pname)
        V[pname] = v_val
        W[pname] = w_val
    for pname in I:
        if pname not in V or pname not in W:
            raise ValueError(f"Missing parameters for product '{pname}'.")
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((V[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((W[i] * x[i] for i in I)) <= C)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')
