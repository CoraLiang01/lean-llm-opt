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
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood.
    -   Distances: from distance.csv, entry for each (school, neighborhood) pair.
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned).
        - Objective: Minimize  sum_{s in S} sum_{n in N} sum_{g in G} [ distance[s, n] * x[s, n, g] ]
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, all students must be assigned to some school:
            sum_{s in S} x[s, n, g] = Population_g[n]   for all n in N, g in G
    -   **School capacity:** For each school s, total assigned students cannot exceed capacity:
            sum_{n in N} sum_{g in G} x[s, n, g] ≤ Capacity[s]   for all s in S
    -   **Racial balance:** For each school s, the percentage of white students assigned must be within 50%–70% of total assigned students:
            0.5 ≤ (sum_{n in N} x[s, n, White]) / (sum_{n in N} sum_{g in G} x[s, n, g]) ≤ 0.7   for all s in S
        (If denominator is zero, i.e., no students assigned, this is infeasible; but with full assignment, denominator will be positive.)
    -   **Non-negativity:** x[s, n, g] ≥ 0 for all s, n, g
[Abstract Model Plan END]