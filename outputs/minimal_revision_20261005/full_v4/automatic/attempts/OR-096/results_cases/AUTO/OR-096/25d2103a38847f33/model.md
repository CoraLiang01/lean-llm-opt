[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from all rows in school_capacity.csv (I, II)
    - Neighborhoods (N): from all rows in neighborhoods_population.csv (N01–N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small enough that continuous is acceptable for planning).
5.  **Identify Parameters (from Schema):**
    -   School capacities: school_capacity.csv, column 'Capacity' for each 'School'
    -   Neighborhood populations: neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each 'Neighborhood'
    -   Distances: distance.csv, entry for each (School, Neighborhood) pair
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned), i.e., minimize sum_{s,n,g} distance[s,n] * x[s,n,g].
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, all students must be assigned: sum_{s} x[s, n, g] = Population_g[n] (where g ∈ {White, NonWhite}).
    -   **School capacity:** For each school s, total assigned students cannot exceed capacity: sum_{n,g} x[s, n, g] ≤ Capacity[s].
    -   **Racial balance:** For each school s, the percentage of white students assigned must be between 50% and 70%:
        - Let total_white[s] = sum_{n} x[s, n, White]
        - Let total_students[s] = sum_{n,g} x[s, n, g]
        - Enforce: 0.5 ≤ total_white[s] / total_students[s] ≤ 0.7 (for total_students[s] > 0; if a school is empty, this is trivially satisfied).
    -   **Non-negativity:** x[s, n, g] ≥ 0 for all s, n, g.
[Abstract Model Plan END]