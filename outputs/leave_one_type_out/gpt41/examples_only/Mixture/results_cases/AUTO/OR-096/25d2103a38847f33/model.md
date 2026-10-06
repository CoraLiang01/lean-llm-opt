[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so that (a) all students are assigned, (b) no school exceeds its capacity, (c) each school’s white-student percentage is within 10 percentage points of the district-wide 60% white/40% nonwhite ratio, and (d) the total student travel distance is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01, ..., N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small so continuous is acceptable for planning).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n).
    -   District-wide racial ratio: 60% white, 40% nonwhite (from query, not schema).
6.  **Formulate Objective:** Minimize the total travel distance for all students, i.e., sum over all schools, neighborhoods, and groups of (distance from school s to neighborhood n) × (number of students of group g assigned from n to s):  
        Minimize:  
        sum_{s in S} sum_{n in N} sum_{g in G} [distance[s, n] * x[s, n, g]]
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, all students must be assigned to some school:  
        sum_{s in S} x[s, n, g] = Population_g[n]  (for g in {White, NonWhite})
    -   **School capacity:** For each school s, total assigned students cannot exceed capacity:  
        sum_{n in N} sum_{g in G} x[s, n, g] ≤ Capacity[s]
    -   **Racial balance:** For each school s, the percentage of white students assigned must be within 10 percentage points of 60% (i.e., between 50% and 70%):  
        0.5 ≤ (sum_{n in N} x[s, n, White]) / (sum_{n in N} sum_{g in G} x[s, n, g]) ≤ 0.7  
        (This is a ratio constraint; in LP, can be linearized by cross-multiplying denominators.)
    -   **Nonnegativity:** x[s, n, g] ≥ 0 for all s, n, g.
[Abstract Model Plan END]