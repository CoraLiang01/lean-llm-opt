[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so that (a) each school's racial composition is within ±10 percentage points of the district's 60% white/40% nonwhite ratio, (b) no school exceeds its capacity, (c) all students are assigned, and (d) the total travel distance for all students is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01, ..., N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, integer if required by context).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood.
    -   Distances: from distance.csv, entry for each (school, neighborhood) pair.
    -   District-wide racial ratio: 60% white, 40% nonwhite (from query, not schema).
6.  **Formulate Objective:** Minimize the total travel distance for all students, i.e., sum over all schools, neighborhoods, and groups of (number of students assigned) × (distance from neighborhood to school).
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, the sum over schools s of x[s, n, g] equals the total number of students of group g in neighborhood n (from neighborhoods_population.csv).
    -   **School capacity:** For each school s, the sum over all neighborhoods n and both groups g of x[s, n, g] ≤ school capacity (from school_capacity.csv).
    -   **Racial balance:** For each school s, the percentage of white students assigned to s must be between 50% and 70% of the total assigned to s (i.e., within ±10% of the district's 60% white ratio). That is, for each school s:
        -   0.5 ≤ (total white students assigned to s) / (total students assigned to s) ≤ 0.7, with appropriate handling if denominator is zero.
    -   **Nonnegativity:** All x[s, n, g] ≥ 0, and (if required) integer.
[Abstract Model Plan END]