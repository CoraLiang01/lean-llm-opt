[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost project crashing model for a renovation project with precedence-linked activities. Each activity can be crashed (shortened) by an integer number of days within specified limits, incurring a per-day crash cost. The model must determine the optimal start times and crash days for each activity to minimize total crashing cost, while respecting activity precedence, crash limits, and a project deadline.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (due to integer crash days and continuous start times).
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of crash days used for activity i (i.e., how many days the activity is shortened). Type: GRB.INTEGER (bounded).
    -   `T` = Project completion time. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   'NormalDuration' (from project_activities.csv): The original duration of activity i.
        -   'CrashDuration' (from project_activities.csv): The minimum possible duration of activity i after crashing.
        -   'CrashCostPerDay' (from project_activities.csv): The cost to crash activity i by one day.
        -   'Predecessors' (from project_activities.csv): List of immediate predecessor activities for i.
    -   'ProjectDeadline' (from project_parameters.csv): The required project completion deadline.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   **Crash Day Bounds:** For each activity i, the number of crash days must be within allowable limits:
        -   Lower bound: z[i] ≥ 0.
        -   Upper bound: z[i] ≤ (NormalDuration[i] - CrashDuration[i]) (i.e., cannot crash more than allowed).
        -   z[i] must be integer.
    -   **Activity Duration Calculation:** The actual duration of activity i is (NormalDuration[i] - z[i]).
    -   **Precedence Constraints:** For each activity i and each predecessor j in Predecessors[i]:
        -   s[i] ≥ s[j] + (NormalDuration[j] - z[j])
        -   (i.e., activity i cannot start until all its predecessors have finished, accounting for their crashed durations.)
    -   **Project Completion Constraints:** For each activity i:
        -   T ≥ s[i] + (NormalDuration[i] - z[i])
        -   (i.e., project completion time is at least the finish time of every activity.)
    -   **Project Deadline Constraint:** T ≤ ProjectDeadline (from project_parameters.csv).
    -   **Nonnegativity:** All start times s[i] ≥ 0, and T ≥ 0.
    -   **Variable Types:** z[i] integer, s[i] and T continuous.
[Abstract Model Plan END]