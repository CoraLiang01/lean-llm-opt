[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so that (a) each school's racial composition is within ±10 percentage points of the district's 60% white/40% nonwhite ratio, (b) no school exceeds its capacity, (c) all students are assigned, and (d) the total travel distance for all students is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from `school_capacity.csv` (I, II)
    - Neighborhoods (N): from `neighborhoods_population.csv` (N01, ..., N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be restricted to integer if required, but not specified in query).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from `school_capacity.csv`, column 'Capacity' for each school.
    -   Neighborhood populations: from `neighborhoods_population.csv`, columns 'Population_White' and 'Population_NonWhite' for each neighborhood.
    -   Distances: from `distance.csv`, entry for each (school, neighborhood) pair.
    -   District-wide white and nonwhite totals: sum over all neighborhoods.
    -   Racial balance target: 60% white, 40% nonwhite; allowed deviation: ±10 percentage points.
6.  **Formulate Objective:** Minimize the total travel distance for all students, i.e., sum over all schools, neighborhoods, and groups of (number of students assigned) × (distance from neighborhood to school).
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood and group, all students must be assigned to some school:
        - For all n in N, for g in G: sum over s in S of x[s, n, g] = Population_g[n]
    -   **School capacity:** For each school, total assigned students (all neighborhoods, both groups) ≤ school capacity:
        - For all s in S: sum over n in N, g in G of x[s, n, g] ≤ Capacity[s]
    -   **Racial balance:** For each school, the percentage of white students assigned must be within [50%, 70%] (i.e., within ±10% of 60%):
        - For all s in S:
            - Let W_s = sum over n of x[s, n, White]
            - Let T_s = sum over n, g of x[s, n, g]
            - Enforce: 0.5 ≤ W_s / T_s ≤ 0.7 (if T_s > 0; if T_s = 0, school is empty and constraint is trivially satisfied)
    -   **Nonnegativity:** All x[s, n, g] ≥ 0
[Abstract Model Plan END]