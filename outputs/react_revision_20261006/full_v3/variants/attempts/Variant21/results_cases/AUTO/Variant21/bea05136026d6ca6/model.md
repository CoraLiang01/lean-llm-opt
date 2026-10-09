[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence relationships and a required project deadline. Each activity can be crashed by an integer number of days within specified bounds, incurring a per-day crash cost. The model must include start times, crash days, and project completion time, with all relevant constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for project crashing with precedence constraints.
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of crash days used for activity i (i.e., how many days the activity is shortened). Type: GRB.INTEGER (bounded).
    -   `T` = Project completion time. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' column in project_activities.csv.
    -   Normal and crash durations: from 'NormalDuration' and 'CrashDuration' columns in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' column in project_activities.csv (semicolon-separated list).
    -   Project deadline: from 'Value' column in project_parameters.csv where 'Parameter' == 'ProjectDeadline'.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Crash bounds): For each activity i, z[i] is integer and satisfies 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   Constraint 2 (Activity duration): The actual duration of activity i is (NormalDuration[i] - z[i]).
    -   Constraint 3 (Precedence): For each activity i and each predecessor j in Predecessors[i], enforce s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Constraint 4 (Project completion): For each activity i, T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Constraint 5 (Project deadline): T ≤ ProjectDeadline (from project_parameters.csv).
    -   Constraint 6 (Nonnegativity): s[i] ≥ 0 for all i; T ≥ 0.
    -   Constraint 7 (Integrality): z[i] are integer for all i.
[Abstract Model Plan END]