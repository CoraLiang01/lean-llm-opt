[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring that (a) no school exceeds its capacity, (b) all students are assigned, and (c) each school’s white-student percentage is within 10 percentage points of the district-wide ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01, ..., N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be restricted to integer if required, but not specified in query).
5.  **Identify Parameters (from Schema):**
    -   School capacities: school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: distance.csv, entry for each (school s, neighborhood n) pair.
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school s to neighborhood n) × (number of students of group g assigned from n to s).
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, the sum over schools s of x[s, n, g] = total number of students of group g in neighborhood n (i.e., all students must be assigned to a school).
    -   **School capacity:** For each school s, the total number of students assigned (sum over all neighborhoods and both groups) ≤ school s’s capacity.
    -   **Racial balance:** For each school s, the percentage of white students assigned must be between 50% and 70% of the total assigned to that school. That is, for each school s:
        -   0.5 ≤ (total white students assigned to s) / (total students assigned to s) ≤ 0.7
        -   In variables: 0.5 × (total students assigned to s) ≤ (total white students assigned to s) ≤ 0.7 × (total students assigned to s)
        -   (If total students assigned to s = 0, this constraint is vacuously satisfied, but in practice, all capacity will be used.)
    -   **Nonnegativity:** All x[s, n, g] ≥ 0.
[Abstract Model Plan END]