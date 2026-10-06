[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from `school_capacity.csv` (Schools I and II)
    - Neighborhoods (N): from `neighborhoods_population.csv` (N01 to N31)
    - Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but not specified).
5.  **Identify Parameters (from Schema):**
    -   School capacities: `school_capacity.csv` columns 'School', 'Capacity' (all rows used).
    -   Neighborhood populations: `neighborhoods_population.csv` columns 'Neighborhood', 'Population_White', 'Population_NonWhite' (all rows used).
    -   Distances: `distance.csv` columns 'School', 'N01'...'N31' (all rows used; join on School and Neighborhood).
    -   District-wide white and nonwhite totals: sum over all neighborhoods.
    -   Racial balance target: 60% white (±10 percentage points).
6.  **Formulate Objective:** Minimize the total distance traveled by all students, i.e., sum over all schools, neighborhoods, and groups of (number of students assigned) × (distance from school to neighborhood):
        Minimize: sum_{s in S} sum_{n in N} sum_{g in G} [ x[s, n, g] × distance[s, n] ]
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood and group, all students must be assigned to some school:
            For all n in N, for g in G:
                sum_{s in S} x[s, n, g] = Population_g[n]
    -   **School Capacity:** For each school, total assigned students cannot exceed capacity:
            For all s in S:
                sum_{n in N} sum_{g in G} x[s, n, g] ≤ Capacity[s]
    -   **Racial Balance:** For each school, the percentage of white students assigned must be between 50% and 70%:
            For all s in S:
                0.5 ≤ (sum_{n in N} x[s, n, White]) / (sum_{n in N} sum_{g in G} x[s, n, g]) ≤ 0.7
        (If denominator is zero, school is empty; but with full assignment, this will not occur.)
    -   **Non-negativity:** All x[s, n, g] ≥ 0
[Abstract Model Plan END]