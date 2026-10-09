[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal crashing plan for a project with multiple activities and precedence relationships, minimizing total crashing cost while ensuring the project completes by a specified deadline. Each activity can be crashed (shortened) by an integer number of days, at a given per-day cost, but not below its crash duration. The model must include start times, integer crashing variables, precedence constraints using crashed durations, project completion time, deadline, and appropriate variable domains.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) project crashing and scheduling problem.
3.  **Define Index Sets:** The primary indices are Activities (set of all activities from project_activities.csv).
4.  **Define Decision Variables:**
    -   `s[i]` = Start time of activity i. Type: GRB.CONTINUOUS (nonnegative).
    -   `z[i]` = Integer number of days by which activity i is crashed (i.e., reduced from normal duration). Type: GRB.INTEGER (bounded).
    -   `T` = Project completion time (makespan). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   For each activity i:
        -   `NormalDuration[i]` (from 'NormalDuration' column in project_activities.csv)
        -   `CrashDuration[i]` (from 'CrashDuration' column)
        -   `CrashCostPerDay[i]` (from 'CrashCostPerDay' column)
        -   `Predecessors[i]` (from 'Predecessors' column; may be empty)
    -   Project deadline: `ProjectDeadline` (from 'Value' in project_parameters.csv where 'Parameter' == 'ProjectDeadline')
6.  **Formulate Objective:** Minimize total crashing cost, i.e., minimize sum over all activities i of (`CrashCostPerDay[i]` * `z[i]`).
7.  **Formulate Constraints:**
    -   Precedence constraints: For each activity i and each predecessor j in `Predecessors[i]`, enforce that `s[i] >= s[j] + (NormalDuration[j] - z[j])`.
    -   Crashing bounds: For each activity i, enforce `0 <= z[i] <= NormalDuration[i] - CrashDuration[i]` and `z[i]` integer.
    -   Start time nonnegativity: For each activity i, `s[i] >= 0`.
    -   Project completion: For each activity i with no successors (i.e., terminal activities), enforce `T >= s[i] + (NormalDuration[i] - z[i])`.
    -   Project deadline: Enforce `T <= ProjectDeadline`.
    -   Variable domains: `s[i]` and `T` are continuous and nonnegative; `z[i]` are integer and within bounds.
[Abstract Model Plan END]