[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost project crashing model for an installation project with precedence-constrained activities. Each activity can be crashed (shortened) by an integer number of days within specified limits, incurring a per-day crash cost. The model must determine the optimal start times and crash days for each activity, ensuring all precedence and deadline constraints are satisfied, and minimizing total crash cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (due to integer crash days and continuous start times).
3.  **Define Index Sets:** The primary indices are Activities (set of all activities listed in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i. Type: GRB.INTEGER, z[i] ≥ 0.
    -   `T` = Project completion time. Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   `NormalDuration[i]` (from 'NormalDuration' column)
        -   `CrashDuration[i]` (from 'CrashDuration' column)
        -   `CrashCostPerDay[i]` (from 'CrashCostPerDay' column)
        -   `Predecessors[i]` (from 'Predecessors' column; may be empty)
    -   Project deadline: `ProjectDeadline` (from project_parameters.csv, 'Value' where 'Parameter' = 'ProjectDeadline')
6.  **Formulate Objective:** Minimize total crash cost, i.e., minimize sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   **Precedence Constraints:** For each activity i and each predecessor j in Predecessors[i]: s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   **Crash Day Bounds:** For each activity i: 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), and z[i] is integer.
    -   **Activity Completion Constraints:** For each activity i: s[i] + (NormalDuration[i] - z[i]) ≤ T.
    -   **Project Deadline Constraint:** T ≤ ProjectDeadline.
    -   **Nonnegativity:** For all i, s[i] ≥ 0; T ≥ 0.
    -   **Integer Restrictions:** For all i, z[i] ∈ {0, 1, ..., NormalDuration[i] - CrashDuration[i]}.
[Abstract Model Plan END]