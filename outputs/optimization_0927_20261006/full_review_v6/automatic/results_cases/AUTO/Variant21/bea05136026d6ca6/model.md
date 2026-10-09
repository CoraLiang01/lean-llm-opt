[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence constraints, crash limits, and a project deadline. The model must decide, for each activity, how many days to crash (integer), when to start (continuous), and ensure all activities finish by the project deadline, while minimizing total crash costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing problem with precedence constraints.
3.  **Define Index Sets:** The primary indices are Activities (from project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i (i.e., how many days its duration is shortened). Type: GRB.INTEGER, with lower and upper bounds per activity.
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' column in project_activities.csv.
    -   Normal and crash durations: from 'NormalDuration' and 'CrashDuration' columns in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' column in project_activities.csv.
    -   Project deadline: from 'Value' in project_parameters.csv where 'Parameter' == 'ProjectDeadline'.
6.  **Formulate Objective:** Minimize total crash cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crash-day bounds: For each activity i, z[i] is integer and satisfies 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   Activity finish constraints: For each activity i, s[i] + (NormalDuration[i] - z[i]) ≤ T.
    -   Project deadline: T ≤ ProjectDeadline (from project_parameters.csv).
    -   Nonnegativity: s[i] ≥ 0 for all i; T ≥ 0.
    -   Integrality: z[i] are integer for all i.
[Abstract Model Plan END]