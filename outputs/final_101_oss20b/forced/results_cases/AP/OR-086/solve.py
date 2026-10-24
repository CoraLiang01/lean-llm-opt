CSVQA_DATA = {'query': 'Develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three distinct wine brands by blending three grades of raw wine material.\n\nThe necessary input data is provided in two files: 30-1.csv contains the daily supply limits and unit costs for the three raw grades, while 30-2.csv provides the selling prices and the blending requirements for all three brands.\n\nThe model should incorporate the following constraints. First, the Blending Requirements constraints specify upper-bound and lower-bound proportion requirements for selected raw grades in each wine brand. Second, the Raw Material Supply constraints require that the total amount of each raw grade used across all brands cannot exceed its corresponding daily supply limit. Third, the Minimum Production constraint requires that the Red brand must have a daily production level of at least 2,000 kg.\n\nThe objective is to determine the quantity of each raw grade allocated to each brand, defined as the decision variables, so as to maximize total net profit, calculated as total sales revenue minus total raw material cost.', 'relationships': [], 'tables': [{'columns': ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'], 'file_name': '30-1.csv', 'records': [{'data_row_number': 0, 'values': {'Cost (CNY/kg)': '6', 'Daily Supply (kg)': '1500', 'Grade': 'I'}}, {'data_row_number': 1, 'values': {'Cost (CNY/kg)': '4.5', 'Daily Supply (kg)': '2000', 'Grade': 'II'}}, {'data_row_number': 2, 'values': {'Cost (CNY/kg)': '3', 'Daily Supply (kg)': '1000', 'Grade': 'III'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'], 'file_name': '30-2.csv', 'records': [{'data_row_number': 0, 'values': {'Blending Requirements': 'I less than 10%  II more than 50%', 'Brand': 'Red', 'Selling Price (CNY/kg)': '5.5'}}, {'data_row_number': 1, 'values': {'Blending Requirements': 'III less than 70%  I more than 20%', 'Brand': 'Yellow', 'Selling Price (CNY/kg)': '5'}}, {'data_row_number': 2, 'values': {'Blending Requirements': 'I less than 50%  II more than 10%', 'Brand': 'Blue', 'Selling Price (CNY/kg)': '4.8'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    G = [rec['values']['Grade'] for rec in table0['records']]
    B = [rec['values']['Brand'] for rec in table1['records']]
    c_g = {}
    s_g = {}
    for rec in table0['records']:
        grade = rec['values']['Grade']
        c_g[grade] = float(rec['values']['Cost (CNY/kg)'])
        s_g[grade] = float(rec['values']['Daily Supply (kg)'])
    p_b = {}
    for rec in table1['records']:
        brand = rec['values']['Brand']
        p_b[brand] = float(rec['values']['Selling Price (CNY/kg)'])
    l_gb = {(g, b): 0.0 for g in G for b in B}
    u_gb = {(g, b): 1.0 for g in G for b in B}
    for rec in table1['records']:
        brand = rec['values']['Brand']
        req_str = rec['values']['Blending Requirements']
        parts = [p.strip() for p in req_str.replace('  ', ' ').split(' ') if p.strip()]
        i = 0
        while i < len(parts):
            grade = parts[i]
            if i + 2 >= len(parts):
                break
            cond = parts[i + 1]
            val = parts[i + 2]
            if val.endswith('%'):
                val = float(val[:-1]) / 100.0
            else:
                val = float(val) / 100.0
            if cond.lower() == 'less':
                u_gb[grade, brand] = val
            elif cond.lower() == 'more':
                l_gb[grade, brand] = val
            i += 3
    for g in G:
        if g not in c_g or g not in s_g:
            raise ValueError(f'Missing cost or supply for grade {g}')
    for b in B:
        if b not in p_b:
            raise ValueError(f'Missing selling price for brand {b}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(g, b) for g in G for b in B], lb=0, vtype=GRB.CONTINUOUS, name='')
    revenue = gurobipy.quicksum((p_b[b] * gurobipy.quicksum((x[g, b] for g in G)) for b in B))
    cost = gurobipy.quicksum((c_g[g] * gurobipy.quicksum((x[g, b] for b in B)) for g in G))
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    for b in B:
        total_b = gurobipy.quicksum((x[g, b] for g in G))
        for g in G:
            model.addConstr(l_gb[g, b] * total_b <= x[g, b])
            model.addConstr(x[g, b] <= u_gb[g, b] * total_b)
    for g in G:
        model.addConstr(gurobipy.quicksum((x[g, b] for b in B)) <= s_g[g])
    model.addConstr(gurobipy.quicksum((x[g, 'Red'] for g in G)) >= 2000)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    return model
m = solve_problem()
