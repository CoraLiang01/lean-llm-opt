[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with multiple activities and precedence relationships, minimizing total crashing cost while ensuring the project completes by a specified deadline. Each activity can be crashed (shortened) by an integer number of days, at a given per-day cost, but not below its crash duration. The model must include start times, integer crashing decisions, precedence constraints using crashed durations, and enforce the project deadline.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Integer number of days by which activity i is crashed (i.e., reduced from normal duration). Type: GRB.INTEGER (bounded).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   Normal duration: project_activities.csv['NormalDuration'][i]
        -   Crash duration: project_activities.csv['CrashDuration'][i]
        -   Crash cost per day: project_activities.csv['CrashCostPerDay'][i]
        -   Immediate predecessors: project_activities.csv['Predecessors'][i] (semicolon-separated list or blank)
    -   Project deadline: project_parameters.csv['Value'] where Parameter == 'ProjectDeadline'
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crashing bounds: For each activity i, 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), and z[i] is integer.
    -   Activity completion: For each activity i, T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline: T ≤ ProjectDeadline (from project_parameters.csv).
    -   Nonnegativity: For all i, s[i] ≥ 0.
    -   Variable domains: z[i] integer, s[i] and T continuous.
[Abstract Model Plan END]