import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    skill_order = {'Junior': 0, 'Intermediate': 1, 'Senior': 2, 'Expert': 3}
    workers_data = [{'worker_id': 'W00', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W01', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W02', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W05', 'skill': 'Senior', 'on_leave': 0}, {'worker_id': 'W10', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W11', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W14', 'skill': 'Expert', 'on_leave': 0}, {'worker_id': 'W16', 'skill': 'Senior', 'on_leave': 0}, {'worker_id': 'W17', 'skill': 'Junior', 'on_leave': 0}, {'worker_id': 'W18', 'skill': 'Intermediate', 'on_leave': 0}, {'worker_id': 'W24', 'skill': 'Expert', 'on_leave': 0}]
    projects_data = [{'project_id': 'P00', 'required_skill': 'Junior'}, {'project_id': 'P01', 'required_skill': 'Junior'}, {'project_id': 'P02', 'required_skill': 'Junior'}, {'project_id': 'P03', 'required_skill': 'Intermediate'}, {'project_id': 'P04', 'required_skill': 'Junior'}, {'project_id': 'P05', 'required_skill': 'Junior'}, {'project_id': 'P06', 'required_skill': 'Junior'}, {'project_id': 'P07', 'required_skill': 'Expert'}, {'project_id': 'P08', 'required_skill': 'Junior'}, {'project_id': 'P09', 'required_skill': 'Intermediate'}]
    cost_matrix = {'W00': {'P00': None, 'P01': 1116, 'P02': 414, 'P03': None, 'P04': 554, 'P05': 113, 'P06': None, 'P07': 11, 'P08': 853, 'P09': 1142}, 'W01': {'P00': None, 'P01': 170, 'P02': None, 'P03': 919, 'P04': 660, 'P05': 1314, 'P06': 668, 'P07': 4, 'P08': 462, 'P09': 614}, 'W02': {'P00': 330, 'P01': 193, 'P02': None, 'P03': 4, 'P04': 431, 'P05': None, 'P06': 1130, 'P07': 15, 'P08': 644, 'P09': 9}, 'W05': {'P00': 325, 'P01': 772, 'P02': 1042, 'P03': None, 'P04': 1394, 'P05': None, 'P06': 374, 'P07': 2, 'P08': 1140, 'P09': 127}, 'W10': {'P00': 1079, 'P01': 1128, 'P02': 758, 'P03': None, 'P04': None, 'P05': 108, 'P06': None, 'P07': 423, 'P08': 744, 'P09': 347}, 'W11': {'P00': 1143, 'P01': 841, 'P02': 920, 'P03': 122, 'P04': 634, 'P05': None, 'P06': 1250, 'P07': 836, 'P08': 180, 'P09': 1105}, 'W14': {'P00': None, 'P01': 687, 'P02': 126, 'P03': 547, 'P04': 125, 'P05': 1358, 'P06': 1391, 'P07': 814, 'P08': 1362, 'P09': 962}, 'W16': {'P00': 265, 'P01': 1114, 'P02': None, 'P03': 1209, 'P04': 231, 'P05': 221, 'P06': None, 'P07': 17, 'P08': 135, 'P09': None}, 'W17': {'P00': 538, 'P01': None, 'P02': 1359, 'P03': 20, 'P04': 1366, 'P05': 702, 'P06': 122, 'P07': 13, 'P08': 358, 'P09': None}, 'W18': {'P00': 238, 'P01': None, 'P02': 1085, 'P03': 529, 'P04': 582, 'P05': 889, 'P06': 139, 'P07': 18, 'P08': 630, 'P09': 538}, 'W24': {'P00': 784, 'P01': 1178, 'P02': 1171, 'P03': None, 'P04': 103, 'P05': 137, 'P06': 832, 'P07': 1279, 'P08': 893, 'P09': 351}}
    workers = [w['worker_id'] for w in workers_data if w['on_leave'] == 0]
    projects = [p['project_id'] for p in projects_data]
    worker_skill = {w['worker_id']: skill_order[w['skill']] for w in workers_data if w['on_leave'] == 0}
    project_required_skill = {p['project_id']: skill_order[p['required_skill']] for p in projects_data}
    eligible_assignments = []
    cost = {}
    for w in workers:
        cost[w] = {}
        for p in projects:
            c = cost_matrix[w][p]
            if c is not None and worker_skill[w] >= project_required_skill[p]:
                eligible_assignments.append((w, p))
                cost[w][p] = c
            else:
                cost[w][p] = None
    W_p = {p: [w for w in workers if cost[w][p] is not None] for p in projects}
    P_w = {w: [p for p in projects if cost[w][p] is not None] for w in workers}
    for p in projects:
        if not W_p[p]:
            raise ValueError(f'No eligible workers for project {p}')
    for w, p in eligible_assignments:
        if cost[w][p] is None:
            raise ValueError(f'Missing cost for eligible assignment ({w}, {p})')
    m = gp.Model('worker_project_assignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(eligible_assignments, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w][p] * x[w, p] for w, p in eligible_assignments)), GRB.MINIMIZE)
    for p in projects:
        m.addConstr(gp.quicksum((x[w, p] for w in W_p[p])) == 1, name=f'project_{p}')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, p] for p in P_w[w])) <= 1, name=f'worker_{w}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()