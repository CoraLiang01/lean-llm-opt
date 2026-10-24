CSVQA_DATA = {'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with these costs detailed in “fixed_cost.csv.” Every supermarket must procure a unit of goods from the available suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in “transportation_costs.csv,” while demand data is available in “demand.csv.” The goal is to identify which suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. Binary decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the share of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the most cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '144'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    demand_table = tables['file_0_view_0']
    customers = [rec['values']['customer'] for rec in demand_table['records']]
    d = {}
    for rec in demand_table['records']:
        key = rec['values']['customer']
        val_str = rec['values']['demand']
        if key in d:
            raise ValueError(f'Duplicate demand entry for customer {key}')
        d[key] = float(val_str)
    fixed_cost_table = tables['file_1_view_0']
    suppliers = [rec['values']['Unnamed: 0'] for rec in fixed_cost_table['records']]
    f = {}
    for rec in fixed_cost_table['records']:
        key = rec['values']['Unnamed: 0']
        val_str = rec['values']['fixed_costs']
        if key in f:
            raise ValueError(f'Duplicate fixed cost entry for supplier {key}')
        f[key] = float(val_str)
    transport_table = tables['file_2_view_0']
    transport_suppliers = set()
    for rec in transport_table['records']:
        transport_suppliers.add(rec['values']['Unnamed: 0'])
    if set(suppliers) != transport_suppliers:
        missing = set(suppliers) - transport_suppliers
        if missing:
            raise ValueError(f'Missing transportation data for suppliers: {missing}')
        extra = transport_suppliers - set(suppliers)
        if extra:
            raise ValueError(f'Unexpected suppliers in transportation data: {extra}')
    c = {}
    for rec in transport_table['records']:
        supplier = rec['values']['Unnamed: 0']
        for customer in customers:
            if customer not in rec['values']:
                raise ValueError(f'Missing cost for supplier {supplier} to customer {customer}')
            val_str = rec['values'][customer]
            c[supplier, customer] = float(val_str)
    for i in suppliers:
        if i not in f:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in customers:
        if j not in d:
            raise ValueError(f'Missing demand for customer {j}')
    for i in suppliers:
        for j in customers:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for supplier {i} to customer {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in suppliers)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in suppliers for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) == d[j], name='demand_' + j)
    for i in suppliers:
        for j in customers:
            m.addConstr(x[i, j] <= d[j] * y[i], name='link_' + i + '_' + j)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
