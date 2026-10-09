[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost project crashing model for an installation project with precedence relationships, where each activity can be crashed (shortened) by an integer number of days within specified limits, incurring a per-day crash cost. The objective is to minimize total crashing cost while meeting a project deadline, with explicit variables for activity start times, crash days, and project completion time.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a project scheduling/crashing model with integer crash decisions).
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of crash days used for activity i. Type: GRB.INTEGER (bounded, nonnegative).
    -   `T` = Project completion time. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: 'CrashCostPerDay' column in project_activities.csv.
    -   Normal and crash durations: 'NormalDuration' and 'CrashDuration' columns in project_activities.csv.
    -   Precedence relationships: 'Predecessors' column in project_activities.csv.
    -   Project deadline: 'Value' column in project_parameters.csv where 'Parameter' == 'ProjectDeadline'.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., minimize sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in Predecessors[i], enforce s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   Crash-day bounds: For each activity i, enforce 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]), with z[i] integer.
    -   Activity completion constraints: For each activity i, enforce T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   Project deadline constraint: T ≤ ProjectDeadline (from project_parameters.csv).
    -   Nonnegativity: For all i, s[i] ≥ 0; T ≥ 0.
    -   Integer restrictions: For all i, z[i] ∈ {0, 1, ..., NormalDuration[i] - CrashDuration[i]}.
[Abstract Model Plan END]