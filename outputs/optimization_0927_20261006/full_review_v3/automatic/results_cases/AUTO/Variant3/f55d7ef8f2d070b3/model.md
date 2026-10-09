[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with multiple activities and precedence relationships, minimizing total crashing cost while ensuring the project completes by a specified deadline. Each activity can be crashed (shortened) by an integer number of days, at a given per-day cost, but not below its crash duration. The model must include start times, integer crashing decisions, precedence constraints using crashed durations, and enforce the project deadline.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (set of all activities in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of days by which activity i is crashed (i.e., reduced from normal duration). Type: GRB.INTEGER (nonnegative).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' column in project_activities.csv.
    -   Normal duration: from 'NormalDuration' column in project_activities.csv.
    -   Crash duration (minimum possible): from 'CrashDuration' column in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' column in project_activities.csv (parsed as sets of predecessor activities).
    -   Project deadline: from 'Value' column in project_parameters.csv where 'Parameter' == 'ProjectDeadline'.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., minimize sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crashing bounds: For each activity i, enforce 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), and z[i] is integer.
    -   Start time nonnegativity: For each activity i, s[i] ≥ 0.
    -   Project completion constraints: For each activity i, T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline constraint: T ≤ ProjectDeadline (from project_parameters.csv).
    -   Variable domains: s[i], T continuous and nonnegative; z[i] integer and within bounds.
[Abstract Model Plan END]