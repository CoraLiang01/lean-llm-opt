CSVQA_DATA = {'query': 'A logistics company manages multiple warehouses responsible for distributing products to various retail stores. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to determine the optimal quantity of products to be shipped from each warehouse to each store, ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while minimizing the total transportation cost.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '45'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '23'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '94'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '92'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '57'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '52'}}, {'data_row_number': 6, 'values': {'customer': 'C7', 'demand': '23'}}, {'data_row_number': 7, 'values': {'customer': 'C8', 'demand': '99'}}, {'data_row_number': 8, 'values': {'customer': 'C9', 'demand': '99'}}, {'data_row_number': 9, 'values': {'customer': 'C10', 'demand': '77'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '127'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '236'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '168'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '115'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'supply_capacity': '280'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'supply_capacity': '179'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'S7', 'supply_capacity': '135'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'S8', 'supply_capacity': '263'}}, {'data_row_number': 8, 'values': {'Unnamed: 0': 'S9', 'supply_capacity': '283'}}, {'data_row_number': 9, 'values': {'Unnamed: 0': 'S10', 'supply_capacity': '476'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '2077.058672521021', 'C10': '0.0', 'C2': '0.0', 'C3': '54.33526480458508', 'C4': '0.0', 'C5': '0.0', 'C6': '36.17284629162332', 'C7': '0.0', 'C8': '0.0', 'C9': '169.33026926588778', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '2077.058672521021', 'C10': '0.0', 'C2': '0.0', 'C3': '1141.0405608962865', 'C4': '0.0', 'C5': '0.0', 'C6': '651.1112332492198', 'C7': '0.0', 'C8': '0.0', 'C9': '8.063346155518467', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '79.9210295982608', 'C10': '474.24509131006675', 'C2': '474.24509131006675', 'C3': '1477.0676289106607', 'C4': '22.583099586193658', 'C5': '474.24509131006675', 'C6': '41.106596962251096', 'C7': '474.24509131006675', 'C8': '474.24509131006675', 'C9': '624.162539502301', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1659.336929105112', 'C10': '1029.6974643797064', 'C2': '57.20541468776147', 'C3': '186.1519048103841', 'C4': '1201.3137084429907', 'C5': '1029.6974643797064', 'C6': '41.82210594495074', 'C7': '57.20541468776147', 'C8': '1201.3137084429907', 'C9': '884.5633870657458', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1297.2567040858307', 'C10': '1399.7932436376784', 'C2': '77.76629131320436', 'C3': '24.26760227579214', 'C4': '1399.7932436376784', 'C5': '77.76629131320436', 'C6': '53.91161728496604', 'C7': '1399.7932436376784', 'C8': '77.76629131320436', 'C9': '1255.115148013589', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '1998.9090658724567', 'C10': '1149.53596749779', 'C2': '985.3165435695341', 'C3': '2.8541686885814643', 'C4': '1149.53596749779', 'C5': '985.3165435695341', 'C6': '730.6923647662475', 'C7': '54.73980797608523', 'C8': '985.3165435695341', 'C9': '46.803102206463265', 'Unnamed: 0': 'S6'}}, {'data_row_number': 6, 'values': {'C1': '1780.3360050180179', 'C10': '0.0', 'C2': '0.0', 'C3': '1141.0405608962865', 'C4': '0.0', 'C5': '0.0', 'C6': '36.17284629162332', 'C7': '0.0', 'C8': '0.0', 'C9': '8.063346155518467', 'Unnamed: 0': 'S7'}}, {'data_row_number': 7, 'values': {'C1': '75.40935896233042', 'C10': '1338.1987290721909', 'C2': '1338.1987290721909', 'C3': '21.391345987195333', 'C4': '74.34437383734394', 'C5': '74.34437383734394', 'C6': '937.3506239061463', 'C7': '1338.1987290721909', 'C8': '1338.1987290721909', 'C9': '1392.1186581383768', 'Unnamed: 0': 'S8'}}, {'data_row_number': 8, 'values': {'C1': '98.90755583433433', 'C10': '0.0', 'C2': '0.0', 'C3': '978.0347664825314', 'C4': '0.0', 'C5': '0.0', 'C6': '651.1112332492198', 'C7': '0.0', 'C8': '0.0', 'C9': '169.33026926588778', 'Unnamed: 0': 'S9'}}, {'data_row_number': 9, 'values': {'C1': '2077.058672521021', 'C10': '0.0', 'C2': '0.0', 'C3': '54.33526480458508', 'C4': '0.0', 'C5': '0.0', 'C6': '36.17284629162332', 'C7': '0.0', 'C8': '0.0', 'C9': '145.1402307993324', 'Unnamed: 0': 'S10'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    table2 = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    C = [rec['values']['customer'] for rec in table0['records']]
    W = [rec['values']['Unnamed: 0'] for rec in table1['records']]
    d_c = {}
    for rec in table0['records']:
        c = rec['values']['customer']
        val = rec['values']['demand']
        d_c[c] = int(float(val))
    s_w = {}
    for rec in table1['records']:
        w = rec['values']['Unnamed: 0']
        val = rec['values']['supply_capacity']
        s_w[w] = int(float(val))
    c_wc = {}
    for rec in table2['records']:
        w = rec['values']['Unnamed: 0']
        c_wc[w] = {}
        for col in table2['columns'][1:]:
            c = col
            val = rec['values'][col]
            c_wc[w][c] = float(val)
    for w in W:
        if w not in s_w:
            raise ValueError(f'Missing supply capacity for warehouse {w}')
    for c in C:
        if c not in d_c:
            raise ValueError(f'Missing demand for customer {c}')
    for w in W:
        for c in C:
            if c not in c_wc[w]:
                raise ValueError(f'Missing cost for warehouse {w} to customer {c}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    keys = [(w, c) for w in W for c in C]
    x = model.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((c_wc[w][c] * x[w, c] for w in W for c in C)), GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[w, c] for w in W)) == d_c[c] for c in C), name='')
    model.addConstrs((gurobipy.quicksum((x[w, c] for c in C)) <= s_w[w] for w in W), name='')
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.Status)
