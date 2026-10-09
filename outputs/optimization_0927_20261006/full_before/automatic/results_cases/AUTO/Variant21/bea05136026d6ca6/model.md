[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to crash (shorten) the durations of project activities, subject to precedence relationships and a required project deadline. Each activity can be crashed by an integer number of days within specified bounds, incurring a per-day crash cost. The model must decide when each activity starts, how much to crash each activity, and ensure all precedence and deadline constraints are satisfied.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for project crashing (time-cost tradeoff in project scheduling).
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of crash days used for activity i. Type: GRB.INTEGER (bounded, nonnegative).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   'NormalDuration' (from project_activities.csv): The original duration of activity i.
        -   'CrashDuration' (from project_activities.csv): The minimum possible duration of activity i after crashing.
        -   'CrashCostPerDay' (from project_activities.csv): The cost to crash activity i by one day.
        -   'Predecessors' (from project_activities.csv): List of immediate predecessor activities for i.
    -   'ProjectDeadline' (from project_parameters.csv): The required project completion deadline.
6.  **Formulate Objective:** Minimize the total crash cost across all activities, i.e., sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   **Crash Bounds:** For each activity i, the number of crash days must be between 0 and (NormalDuration[i] - CrashDuration[i]), i.e., 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]).
    -   **Precedence Constraints:** For each activity i and each predecessor j in Predecessors[i], ensure that activity i cannot start until all its predecessors have finished, i.e., s[i] ≥ s[j] + (NormalDuration[j] - z[j]).
    -   **Completion Time Constraints:** For each activity i, the project completion time T must be at least the finish time of i, i.e., T ≥ s[i] + (NormalDuration[i] - z[i]).
    -   **Project Deadline Constraint:** The project must finish by the required deadline, i.e., T ≤ ProjectDeadline.
    -   **Nonnegativity:** All start times s[i] and the project completion time T must be ≥ 0.
    -   **Integrality:** All crash variables z[i] must be integer and within their bounds.
[Abstract Model Plan END]