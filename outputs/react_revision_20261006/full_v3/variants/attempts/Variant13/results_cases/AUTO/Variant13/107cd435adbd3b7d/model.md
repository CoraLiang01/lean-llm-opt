[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost project crashing model for an installation project with activity precedence, where each activity can be crashed (shortened) by an integer number of days within specified limits, incurring a per-day crash cost. The goal is to minimize total crash costs while ensuring all precedence and deadline constraints are met.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (due to integer crash variables and continuous time variables).
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS, s[i] ≥ 0.
    -   `z[i]` = Number of crash days used for activity i (integer, between 0 and NormalDuration - CrashDuration). Type: GRB.INTEGER, z[i] ≥ 0.
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS, T ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' column in project_activities.csv.
    -   Normal and crash durations: from 'NormalDuration' and 'CrashDuration' columns in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' column in project_activities.csv (semicolon-separated list).
    -   Project deadline: from 'Value' column in project_parameters.csv where Parameter == 'ProjectDeadline'.
6.  **Formulate Objective:** Minimize total crash cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   **Precedence Constraints:** For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]). This ensures that an activity cannot start until all its predecessors are finished, accounting for any crashing.
    -   **Crash Day Bounds:** For each activity i, enforce 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), and z[i] is integer. This ensures crashing does not exceed feasible limits.
    -   **Project Completion Constraints:** For each activity i with no successors (i.e., terminal activities), enforce T ≥ s[i] + (NormalDuration[i] - z[i]). This ensures T is at least as large as the finish time of the last activity.
    -   **Project Deadline Constraint:** Enforce T ≤ ProjectDeadline (from project_parameters.csv).
    -   **Nonnegativity:** s[i] ≥ 0 for all activities; T ≥ 0.
    -   **Integrality:** z[i] are integer variables.
[Abstract Model Plan END]