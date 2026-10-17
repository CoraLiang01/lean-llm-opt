[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence constraints, crash limits, and a project deadline. The model must decide, for each activity, how many days to crash (integer), when to start (continuous), and ensure all activities finish by the project deadline, while minimizing total crash costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for project crashing with precedence and deadline constraints.
3.  **Define Index Sets:** The primary indices are Activities (from project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i (integer, between 0 and NormalDuration[i] - CrashDuration[i]). Type: GRB.INTEGER.
    -   `T` = Project completion time (maximum finish time over all activities). Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: 'CrashCostPerDay' (project_activities.csv).
    -   Normal and crash durations: 'NormalDuration', 'CrashDuration' (project_activities.csv).
    -   Precedence relationships: 'Predecessors' (project_activities.csv).
    -   Project deadline: 'ProjectDeadline' (project_parameters.csv).
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Precedence): For each activity i and each predecessor j in Predecessors[i], enforce s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Constraint 2 (Crash-day bounds): For each activity i, enforce 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), with z[i] integer.
    -   Constraint 3 (Completion time): For each activity i, enforce s[i] + (NormalDuration[i] - z[i]) ≤ T.
    -   Constraint 4 (Project deadline): Enforce T ≤ ProjectDeadline.
    -   Constraint 5 (Nonnegativity): For all i, s[i] ≥ 0; T ≥ 0.
    -   Constraint 6 (Integrality): For all i, z[i] ∈ {0, 1, ..., NormalDuration[i] - CrashDuration[i]}.
[Abstract Model Plan END]