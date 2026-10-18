[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost project crashing model for an installation project with precedence-constrained activities. Each activity can be crashed (shortened) by an integer number of days within specified limits, incurring a per-day crash cost. The model must determine the optimal start times and crash days for each activity, ensuring all precedence, duration, and deadline constraints are satisfied, and minimizing total crash cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (due to integer crash days and continuous start times).
3.  **Define Index Sets:** The primary indices are Activities (set of all activities listed in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i (integer, between 0 and NormalDuration[i] - CrashDuration[i]). Type: GRB.INTEGER, z[i] ≥ 0.
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' column in project_activities.csv.
    -   Normal and crash durations: from 'NormalDuration' and 'CrashDuration' columns in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' column in project_activities.csv (semicolon-separated list).
    -   Project deadline: from 'Value' column in project_parameters.csv where Parameter = 'ProjectDeadline'.
6.  **Formulate Objective:** Minimize total crash cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crash-day bounds: For each activity i, 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), and z[i] is integer.
    -   Activity completion constraints: For each activity i, T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline constraint: T ≤ ProjectDeadline (from project_parameters.csv).
    -   Nonnegativity: For all i, s[i] ≥ 0; T ≥ 0.
    -   Integer restrictions: For all i, z[i] ∈ {0, 1, ..., NormalDuration[i] - CrashDuration[i]}.
[Abstract Model Plan END]