[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with precedence-constrained activities, where each activity can be shortened (crashed) at a cost, to minimize total crashing cost while meeting a project deadline. The model must decide start times and crash amounts for each activity, respecting precedence, crash limits, and deadline.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (from project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of days by which activity i is crashed (i.e., reduced from normal duration). Type: GRB.INTEGER (nonnegative).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   `NormalDuration[i]` (from 'NormalDuration' column).
        -   `CrashDuration[i]` (from 'CrashDuration' column).
        -   `CrashCostPerDay[i]` (from 'CrashCostPerDay' column).
        -   `Predecessors[i]` (from 'Predecessors' column; may be empty).
    -   Project deadline: `ProjectDeadline` (from project_parameters.csv, 'Value' where 'Parameter' == 'ProjectDeadline').
6.  **Formulate Objective:** Minimize total crashing cost, i.e., minimize sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crash bounds: For each activity i, enforce 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   Project completion constraints: For each activity i, enforce T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline constraint: T ≤ ProjectDeadline.
    -   Nonnegativity: For all i, s[i] ≥ 0; T ≥ 0.
    -   Integer restrictions: For all i, z[i] are integer variables.
[Abstract Model Plan END]