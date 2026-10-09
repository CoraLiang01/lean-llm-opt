[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence relationships, crash limits, and a required project deadline. The model must decide, for each activity, how many days to crash (integer), when to start (continuous), and ensure all activities finish by the project deadline, while minimizing total crash costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing problem.
3.  **Define Index Sets:** The primary indices are Activities (from project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i. Type: GRB.INTEGER, z[i] ≥ 0.
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: 'CrashCostPerDay' column in project_activities.csv.
    -   Normal and crash durations: 'NormalDuration' and 'CrashDuration' columns in project_activities.csv.
    -   Precedence relationships: 'Predecessors' column in project_activities.csv.
    -   Project deadline: 'Value' where 'Parameter' == 'ProjectDeadline' in project_parameters.csv.
6.  **Formulate Objective:** Minimize total crash cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Crash-day bounds): For each activity i, z[i] ≥ 0 (integer), and z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   Constraint 2 (Activity duration): The actual duration of activity i is (NormalDuration[i] - z[i]).
    -   Constraint 3 (Precedence): For each activity i and each predecessor j in Predecessors[i], s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Constraint 4 (Project completion): For each activity i, T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Constraint 5 (Project deadline): T ≤ ProjectDeadline.
    -   Constraint 6 (Nonnegativity): s[i] ≥ 0 for all i; T ≥ 0.
    -   Constraint 7 (Integrality): z[i] are integer for all i.
[Abstract Model Plan END]