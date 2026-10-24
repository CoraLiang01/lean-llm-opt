[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with multiple activities and precedence relationships, minimizing total crashing cost while ensuring the project completes by a specified deadline. Each activity can be crashed (shortened) by an integer number of days, at a per-day cost, but not below its crash duration. The model must include start times, precedence constraints (using crashed durations), bounds on crashing, project completion time, deadline, and variable domains.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Integer number of days by which activity i is crashed (i.e., reduction from normal duration). Type: GRB.INTEGER (bounded).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   'NormalDuration' (project_activities.csv): Normal duration of each activity.
    -   'CrashDuration' (project_activities.csv): Minimum possible duration after crashing.
    -   'CrashCostPerDay' (project_activities.csv): Cost per day to crash each activity.
    -   'Predecessors' (project_activities.csv): Immediate predecessors for each activity (semicolon-separated).
    -   'ProjectDeadline' (project_parameters.csv): Maximum allowed project completion time.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crashing bounds: For each activity i, 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), and z[i] is integer.
    -   Activity finish constraints: For each activity i, s[i] + (NormalDuration[i] - z[i]) ≤ T.
    -   Project deadline: T ≤ ProjectDeadline (from project_parameters.csv).
    -   Nonnegativity: For all i, s[i] ≥ 0.
    -   Variable domains: z[i] integer, s[i] and T continuous and nonnegative.
[Abstract Model Plan END]