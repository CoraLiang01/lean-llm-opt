[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring that (a) no school exceeds its capacity, (b) all students are assigned, and (c) each school’s white-student percentage is within 10 percentage points of the district-wide ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01, ..., N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but not specified).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n) pair.
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
    -   Racial balance bounds: 60% ± 10% (i.e., 50% ≤ %white ≤ 70% at each school).
6.  **Formulate Objective:** Minimize the total travel distance for all students:
        sum over all schools s, neighborhoods n, and groups g of [distance from s to n] × x[s, n, g].
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, all students must be assigned:
            sum over schools s of x[s, n, g] = Population_g[n]   (for g in {White, NonWhite})
    -   **School capacity:** For each school s, total assigned students cannot exceed capacity:
            sum over neighborhoods n and groups g of x[s, n, g] ≤ Capacity[s]
    -   **Racial balance at each school:** For each school s, the percentage of white students assigned must be between 50% and 70%:
            0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7
        (If denominator is zero, the school is empty; but with full assignment, this will not occur.)
    -   **Nonnegativity:** x[s, n, g] ≥ 0 for all s, n, g.
[Abstract Model Plan END]