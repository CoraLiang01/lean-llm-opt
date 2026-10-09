[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01, ..., N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small enough that continuous is acceptable for planning).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n).
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school s to neighborhood n) × (number of students of group g assigned from n to s).
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, the sum over schools s of x[s, n, g] = total number of students of group g in neighborhood n (i.e., all students are assigned to some school).
    -   **School capacity:** For each school s, the sum over all neighborhoods n and both groups g of x[s, n, g] ≤ school s's capacity.
    -   **Racial balance:** For each school s, the percentage of white students assigned to s must be between 50% and 70%:
        - Let W_s = sum over n of x[s, n, White]
        - Let T_s = sum over n and g of x[s, n, g]
        - Enforce: 0.5 × T_s ≤ W_s ≤ 0.7 × T_s
    -   **Non-negativity:** All x[s, n, g] ≥ 0.
[Abstract Model Plan END]