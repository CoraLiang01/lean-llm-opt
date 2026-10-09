[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost project crashing model for a renovation project with precedence-linked activities. Each activity can be crashed (shortened) by an integer number of days within specified limits, incurring a per-day crash cost. The model must determine the start times and crash days for each activity, ensuring all precedence and deadline constraints are satisfied, and minimizing total crash cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a project scheduling/crashing model with integer crash variables and continuous time variables).
3.  **Define Index Sets:** The primary indices are Activities (from the 'Activity' column in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of crash days used for activity i. Type: GRB.INTEGER (bounded).
    -   `T` = Project completion time. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Crash cost per day: from 'CrashCostPerDay' column in project_activities.csv.
    -   Normal duration: from 'NormalDuration' column in project_activities.csv.
    -   Crash duration: from 'CrashDuration' column in project_activities.csv.
    -   Precedence relationships: from 'Predecessors' column in project_activities.csv.
    -   Project deadline: from 'Value' column in project_parameters.csv (where 'Parameter' == 'ProjectDeadline').
6.  **Formulate Objective:** Minimize total crash cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   **Precedence Constraints:** For each activity i and each predecessor j in Predecessors[i]:  
        s[i] ≥ s[j] + (NormalDuration[j] - z[j])  
        (The start of i must be after the finish of each predecessor j, using the crashed duration for j.)
    -   **Crash Day Bounds:** For each activity i:  
        0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i])  
        (Crash days must be integer, nonnegative, and cannot exceed the maximum possible crash for that activity.)
    -   **Completion Time Constraints:** For each activity i:  
        T ≥ s[i] + (NormalDuration[i] - z[i])  
        (Project completion time must be at least the finish time of every activity.)
    -   **Project Deadline Constraint:**  
        T ≤ ProjectDeadline  
        (Project must finish by the required deadline.)
    -   **Nonnegativity and Integrality:**  
        For all i: s[i] ≥ 0 (start times nonnegative), z[i] integer (crash days integer), T ≥ 0.
[Abstract Model Plan END]