[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence relationships and a required project deadline. Each activity can be crashed by an integer number of days within specified bounds, incurring a per-day crash cost. The model must include start times, crash days, and project completion time, with all relevant constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for project crashing with precedence and deadline constraints.
3.  **Define Index Sets:** The primary index is the set of Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of crash days used for activity i (integer, between 0 and NormalDuration - CrashDuration). Type: GRB.INTEGER.
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   'NormalDuration' (from project_activities.csv): The standard duration of activity i.
        -   'CrashDuration' (from project_activities.csv): The minimum possible duration of activity i after crashing.
        -   'CrashCostPerDay' (from project_activities.csv): The cost to crash activity i by one day.
        -   'Predecessors' (from project_activities.csv): List of immediate predecessor activities for i.
    -   'ProjectDeadline' (from project_parameters.csv): The required project completion deadline.
6.  **Formulate Objective:** Minimize the total crash cost across all activities, i.e., sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   **Crash Day Bounds:** For each activity i, z[i] must be integer and satisfy 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   **Precedence Constraints:** For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   **Activity Completion Constraints:** For each activity i, the finish time is s[i] + (NormalDuration[i] - z[i]). For activities with no successors (project finishers), T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   **Project Deadline Constraint:** T ≤ ProjectDeadline (from project_parameters.csv).
    -   **Nonnegativity:** All start times s[i] ≥ 0, and T ≥ 0.
    -   **Integrality:** All z[i] are integer variables.
[Abstract Model Plan END]