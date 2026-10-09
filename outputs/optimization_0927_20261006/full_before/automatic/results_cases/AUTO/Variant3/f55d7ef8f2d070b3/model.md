[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a mixed-integer project crashing model for a construction project with multiple activities, each of which can be crashed (shortened) at a cost, subject to precedence relationships and a project deadline. The goal is to minimize total crashing cost while ensuring all activities are scheduled feasibly and the project completes by the deadline.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for project scheduling with crashing (time-cost tradeoff).
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of days by which activity i is crashed (i.e., reduced from its normal duration). Type: GRB.INTEGER (nonnegative).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   `NormalDuration[i]` (from 'NormalDuration' column): The standard duration of activity i.
        -   `CrashDuration[i]` (from 'CrashDuration' column): The minimum possible duration of activity i after crashing.
        -   `CrashCostPerDay[i]` (from 'CrashCostPerDay' column): The cost to reduce activity i by one day.
        -   `Predecessors[i]` (from 'Predecessors' column): List of immediate predecessor activities for i (may be empty).
    -   Project deadline: `ProjectDeadline` (from 'Value' column in project_parameters.csv, where 'Parameter' == 'ProjectDeadline').
6.  **Formulate Objective:** Minimize total crashing cost, i.e., minimize sum over all activities i of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   **Precedence Constraints:** For each activity i and each predecessor j in Predecessors[i], ensure that activity i cannot start until all its predecessors have finished, using the crashed duration:
        -   s[i] ≥ s[j] + (NormalDuration[j] - z[j])
    -   **Crashing Bounds:** For each activity i, the amount crashed must be between 0 and the maximum possible (NormalDuration[i] - CrashDuration[i]):
        -   0 ≤ z[i] ≤ NormalDuration[i] - CrashDuration[i], and z[i] is integer.
    -   **Project Completion Constraints:** For each activity i, ensure that the project completion time T is at least the finish time of activity i:
        -   T ≥ s[i] + (NormalDuration[i] - z[i])
    -   **Project Deadline Constraint:** The project must finish by the specified deadline:
        -   T ≤ ProjectDeadline
    -   **Nonnegativity Constraints:** For all activities i, s[i] ≥ 0.
    -   **Variable Types:** z[i] are integer, s[i] and T are continuous.
[Abstract Model Plan END]