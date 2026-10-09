[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with multiple activities and precedence relationships, minimizing total crashing cost while ensuring the project completes by a specified deadline. Each activity can be crashed (shortened) by an integer number of days, at a given per-day cost, but not below its crash duration. The model must include start times, precedence constraints (using crashed durations), bounds on crashing, project completion time, deadline, and variable domains.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (set of all activities in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of days by which activity i is crashed (i.e., reduced from normal duration). Type: GRB.INTEGER (nonnegative).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   `NormalDuration[i]` (from 'NormalDuration' column in project_activities.csv): the standard duration.
        -   `CrashDuration[i]` (from 'CrashDuration'): the minimum possible duration after crashing.
        -   `CrashCostPerDay[i]` (from 'CrashCostPerDay'): cost per day of crashing.
        -   `Predecessors[i]` (from 'Predecessors'): list of immediate predecessor activities.
    -   `ProjectDeadline` (from 'Value' in project_parameters.csv where 'Parameter' == 'ProjectDeadline'): the required project completion time.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., minimize sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crashing bounds: For each activity i, 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   Project completion constraints: For each activity i, T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline constraint: T ≤ ProjectDeadline.
    -   Nonnegativity: For all i, s[i] ≥ 0; T ≥ 0.
    -   Integer restrictions: For all i, z[i] ∈ {0, 1, ..., NormalDuration[i] - CrashDuration[i]} (i.e., integer and within bounds).
[Abstract Model Plan END]