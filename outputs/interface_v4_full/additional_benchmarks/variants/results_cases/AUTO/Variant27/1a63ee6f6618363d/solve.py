import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    skill_level = {'Junior': 0, 'Intermediate': 1, 'Senior': 2, 'Expert': 3}
    workers = {'W00': 'Intermediate', 'W01': 'Intermediate', 'W02': 'Junior', 'W05': 'Senior', 'W10': 'Expert', 'W11': 'Expert', 'W14': 'Expert', 'W16': 'Senior', 'W17': 'Junior', 'W18': 'Intermediate', 'W24': 'Expert'}
    projects = {'P00': 'Junior', 'P01': 'Junior', 'P02': 'Junior', 'P03': 'Intermediate', 'P04': 'Junior', 'P05': 'Junior', 'P06': 'Junior', 'P07': 'Expert', 'P08': 'Junior', 'P09': 'Intermediate'}
    cost = {'W00': {'P01': 1116, 'P02': 414, 'P04': 554, 'P05': 113, 'P07': 11, 'P08': 853, 'P09': 1142}, 'W01': {'P01': 170, 'P03': 919, 'P04': 660, 'P05': 1314, 'P06': 668, 'P07': 4, 'P08': 462, 'P09': 614}, 'W02': {'P00': 330, 'P01': 193, 'P03': 4, 'P04': 431, 'P06': 1130, 'P07': 15, 'P08': 644, 'P09': 9}, 'W05': {'P00': 325, 'P01': 772, 'P02': 1042, 'P04': 1394, 'P06': 374, 'P07': 2, 'P08': 1140, 'P09': 127}, 'W10': {'P00': 1079, 'P01': 1128, 'P02': 758, 'P05': 108, 'P07': 423, 'P08': 744, 'P09': 347}, 'W11': {'P00': 1143, 'P01': 841, 'P02': 920, 'P03': 122, 'P04': 634, 'P06': 1250, 'P07': 836, 'P08': 180, 'P09': 1105}, 'W14': {'P01': 687, 'P02': 126, 'P03': 547, 'P04': 125, 'P05': 1358, 'P06': 1391, 'P07': 814, 'P08': 1362, 'P09': 962}, 'W16': {'P00': 265, 'P01': 1114, 'P03': 1209, 'P04': 231, 'P05': 221, 'P07': 17, 'P08': 135}, 'W17': {'P00': 538, 'P02': 1359, 'P03': 20, 'P04': 1366, 'P05': 702, 'P06': 122, 'P07': 13, 'P08': 358}, 'W18': {'P00': 238, 'P02': 1085, 'P03': 529, 'P04': 582, 'P05': 889, 'P06': 139, 'P07': 18, 'P08': 630, 'P09': 538}, 'W24': {'P00': 784, 'P01': 1178, 'P02': 1171, 'P04': 103, 'P05': 137, 'P06': 832, 'P07': 1279, 'P08': 893, 'P09': 351}}
    worker_ids = list(workers.keys())
    project_ids = list(projects.keys())
    allowed_assignments = []
    for i in worker_ids:
        for j in project_ids:
            if j in cost.get(i, {}):
                if skill_level[workers[i]] >= skill_level[projects[j]]:
                    allowed_assignments.append((i, j))
                else:
                    raise ValueError(f'Worker {i} skill {workers[i]} insufficient for project {j} requiring {projects[j]}')
    W_j = {j: [i for i in worker_ids if (i, j) in allowed_assignments] for j in project_ids}
    P_i = {i: [j for j in project_ids if (i, j) in allowed_assignments] for i in worker_ids}
    for j in project_ids:
        if not W_j[j]:
            raise ValueError(f'No eligible worker for project {j}')
    m = gp.Model('worker_project_assignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(allowed_assignments, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i, j in allowed_assignments)), GRB.MINIMIZE)
    for j in project_ids:
        m.addConstr(gp.quicksum((x[i, j] for i in W_j[j])) == 1, name='pj_' + j)
    for i in worker_ids:
        if P_i[i]:
            m.addConstr(gp.quicksum((x[i, j] for j in P_i[i])) <= 1, name='wi_' + i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()