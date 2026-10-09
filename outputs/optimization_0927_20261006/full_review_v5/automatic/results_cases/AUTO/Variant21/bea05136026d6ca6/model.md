[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence constraints, crash limits, and a project deadline. The model must decide, for each activity, how many days to crash (integer), when to start (continuous), and ensure all activities finish by the project deadline, while minimizing total crash costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing problem.
3.  **Define Index Sets:** The primary indices are Activities (from project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i. Type: GRB.INTEGER, z[i] ≥ 0.
    -   `T` = Project completion time. Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' column in project_activities.csv.
    -   Normal and crash durations: from 'NormalDuration' and 'CrashDuration' columns in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' column in project_activities.csv.
    -   Project deadline: from 'Value' where 'Parameter' = 'ProjectDeadline' in project_parameters.csv.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crash-day bounds: For each activity i, enforce 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), with z[i] integer.
    -   Activity completion: For each activity i, enforce T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline: Enforce T ≤ ProjectDeadline.
    -   Nonnegativity: For all i, s[i] ≥ 0; T ≥ 0.
    -   Integrality: For all i, z[i] ∈ {0, 1, ..., NormalDuration[i] - CrashDuration[i]}.
[Abstract Model Plan END]