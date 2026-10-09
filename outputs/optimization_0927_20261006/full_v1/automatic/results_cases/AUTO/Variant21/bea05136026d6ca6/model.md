[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence relationships, crash limits, and a project deadline. The model must decide, for each activity, how many days to crash (integer), when to start (continuous), and ensure all activities finish by the project deadline, while minimizing total crash costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing problem with precedence constraints.
3.  **Define Index Sets:** The primary indices are Activities (set of all activities listed in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i (i.e., how many days the duration is shortened). Type: GRB.INTEGER, with lower and upper bounds per activity.
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   'NormalDuration' (from project_activities.csv): maximum (uncrashed) duration.
        -   'CrashDuration' (from project_activities.csv): minimum (fully crashed) duration.
        -   'CrashCostPerDay' (from project_activities.csv): cost per day of crashing.
        -   'Predecessors' (from project_activities.csv): list of immediate predecessor activities.
    -   'ProjectDeadline' (from project_parameters.csv): required project completion time.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Precedence): For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Constraint 2 (Crash-day bounds): For each activity i, z[i] is integer and satisfies 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   Constraint 3 (Activity completion): For each activity i, the finish time is s[i] + (NormalDuration[i] - z[i]); require that T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Constraint 4 (Project deadline): T ≤ ProjectDeadline (from project_parameters.csv).
    -   Constraint 5 (Nonnegativity): s[i] ≥ 0 for all activities; T ≥ 0.
    -   Constraint 6 (Integrality): z[i] are integer variables for all activities.
[Abstract Model Plan END]