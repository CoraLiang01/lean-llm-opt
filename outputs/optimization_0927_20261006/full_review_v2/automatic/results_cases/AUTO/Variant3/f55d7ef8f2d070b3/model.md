[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with multiple activities and precedence relationships, minimizing total crashing cost while ensuring the project completes by a specified deadline. Each activity can be crashed (shortened) by an integer number of days, at a given per-day cost, but not below its crash duration.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Integer number of days by which activity i is crashed (i.e., duration reduction). Type: GRB.INTEGER (nonnegative).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' in project_activities.csv.
    -   Normal and crash durations: from 'NormalDuration' and 'CrashDuration' in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' in project_activities.csv (parsed as lists).
    -   Project deadline: from 'Value' where 'Parameter' == 'ProjectDeadline' in project_parameters.csv.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crashing bounds: For each activity i, 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), and z[i] is integer.
    -   Activity completion constraints: For each activity i, T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline constraint: T ≤ ProjectDeadline (from project_parameters.csv).
    -   Nonnegativity: For all i, s[i] ≥ 0; T ≥ 0.
    -   Integer restrictions: For all i, z[i] ∈ {0, 1, ..., NormalDuration[i] - CrashDuration[i]}.
[Abstract Model Plan END]