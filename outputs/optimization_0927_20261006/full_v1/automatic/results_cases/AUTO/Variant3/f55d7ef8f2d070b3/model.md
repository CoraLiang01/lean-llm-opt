[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with multiple activities and precedence relationships, minimizing total crashing cost while ensuring the project completes by a specified deadline. Each activity can be crashed (shortened) by paying a per-day cost, but not below its crash duration. The model must include start times, integer crashing decisions, precedence constraints, and deadline enforcement.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (set of all activities listed in project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Number of days by which activity i is crashed (i.e., reduced from normal duration). Type: GRB.INTEGER (nonnegative).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'CrashCostPerDay' (from project_activities.csv) gives the cost per day of crashing each activity.
    -   Constraint coefficients: 'NormalDuration' and 'CrashDuration' (from project_activities.csv) define the allowable range for each activity's duration; 'Predecessors' defines precedence relationships.
    -   Constraint RHS: 'ProjectDeadline' (from project_parameters.csv) gives the maximum allowed project completion time.
6.  **Formulate Objective:** Minimize total crashing cost, i.e., sum over all activities of (CrashCostPerDay[i] * z[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Precedence): For each activity i and each predecessor j in Predecessors[i], enforce that s[i] ≥ s[j] + (NormalDuration[j] - z[j]). This ensures activities cannot start until all predecessors finish, accounting for any crashing.
    -   Constraint 2 (Crashing Bounds): For each activity i, enforce 0 ≤ z[i] ≤ (NormalDuration[i] - CrashDuration[i]). This ensures activities are not crashed below their minimum allowed duration.
    -   Constraint 3 (Completion Time): For each activity i, enforce T ≥ s[i] + (NormalDuration[i] - z[i]). This ensures T is at least the finish time of every activity.
    -   Constraint 4 (Project Deadline): Enforce T ≤ ProjectDeadline (from project_parameters.csv).
    -   Constraint 5 (Nonnegativity): For all i, s[i] ≥ 0.
    -   Constraint 6 (Integrality): For all i, z[i] are integer variables.
[Abstract Model Plan END]