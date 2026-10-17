[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring: (a) each school's total assigned students do not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and proportional/racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from `school_capacity.csv` (I, II)
    - Neighborhoods (N): from `neighborhoods_population.csv` (N01, ..., N31)
    - Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but not specified).
5.  **Identify Parameters (from Schema):**
    -   `Capacity[s]`: School capacity from `school_capacity.csv` (field: Capacity).
    -   `Pop[n, g]`: Number of students of group g in neighborhood n from `neighborhoods_population.csv` (fields: Population_White, Population_NonWhite).
    -   `Dist[s, n]`: Distance in miles from school s to neighborhood n from `distance.csv`.
    -   District-wide white ratio: 60% (from query).
6.  **Formulate Objective:** Minimize the total distance students travel: sum over all schools, neighborhoods, and groups of `Dist[s, n] * x[s, n, g]`.
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood n and group g, the sum over schools s of `x[s, n, g]` equals `Pop[n, g]` (all students assigned).
    -   **School Capacity:** For each school s, the sum over all neighborhoods n and groups g of `x[s, n, g]` is less than or equal to `Capacity[s]`.
    -   **Racial Balance:** For each school s, the percentage of white students assigned must be between 50% and 70%:
        - Lower bound: sum over n of `x[s, n, White]` ≥ 0.5 × total assigned to s
        - Upper bound: sum over n of `x[s, n, White]` ≤ 0.7 × total assigned to s
    -   **Nonnegativity:** All `x[s, n, g]` ≥ 0.
[Abstract Model Plan END]