CSVQA_DATA = {'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with these costs detailed in “fixed_cost.csv.” Every supermarket must procure a unit of goods from the available suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in “transportation_costs.csv,” while demand data is available in “demand.csv.” The goal is to identify which suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. Binary decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the share of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the most cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '144'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table_records = {t['table_id']: t['records'] for t in tables}
    fixed_table = table_records['file_1_view_0']
    suppliers = [rec['values']['Unnamed: 0'] for rec in fixed_table]
    f = {}
    for rec in fixed_table:
        sid = rec['values']['Unnamed: 0']
        try:
            f[sid] = float(rec['values']['fixed_costs'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid fixed cost for supplier {sid}')
    demand_table = table_records['file_0_view_0']
    customers = [rec['values']['customer'] for rec in demand_table]
    d = {}
    for rec in demand_table:
        cid = rec['values']['customer']
        try:
            d[cid] = float(rec['values']['demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid demand for customer {cid}')
    cost_table = table_records['file_2_view_0']
    c = {}
    for rec in cost_table:
        sid = rec['values']['Unnamed: 0']
        if sid not in suppliers:
            raise ValueError(f'Supplier {sid} in cost table not in supplier list')
        c[sid] = {}
        for cid in customers:
            if cid not in rec['values']:
                raise ValueError(f'Missing cost for supplier {sid} to customer {cid}')
            try:
                c[sid][cid] = float(rec['values'][cid])
            except ValueError:
                raise ValueError(f'Invalid cost for supplier {sid} to customer {cid}')
    for sid in suppliers:
        if sid not in f:
            raise ValueError(f'Fixed cost missing for supplier {sid}')
        if sid not in c:
            raise ValueError(f'Cost matrix missing for supplier {sid}')
        for cid in customers:
            if cid not in c[sid]:
                raise ValueError(f'Cost missing for supplier {sid} to customer {cid}')
    for cid in customers:
        if cid not in d:
            raise ValueError(f'Demand missing for customer {cid}')
    m = gurobipy.Model('SupplierActivation')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in suppliers)) + gurobipy.quicksum((c[i][j] * x[i, j] for i in suppliers for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) == d[j], name='')
    for i in suppliers:
        for j in customers:
            m.addConstr(x[i, j] <= d[j] * y[i], name='')
    m.optimize()
    return m
m = solve_problem()
