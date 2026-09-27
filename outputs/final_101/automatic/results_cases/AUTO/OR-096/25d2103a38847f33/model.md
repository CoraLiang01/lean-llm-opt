[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from school_capacity.csv): S = {I, II}
    - Neighborhoods (from neighborhoods_population.csv): N = {N01, N02, ..., N31}
    - Student groups: G = {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small enough that continuous is acceptable for planning).
5.  **Identify Parameters (from Schema):**
    -   School capacities: school_capacity.csv, column 'Capacity' for each 'School' s.
    -   Neighborhood populations: neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each 'Neighborhood' n.
    -   Distances: distance.csv, entry for each (School s, Neighborhood n) pair.
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
6.  **Formulate Objective:** Minimize the total travel distance for all students:  
        sum over all s, n, g of (distance from school s to neighborhood n) × x[s, n, g].
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, all students must be assigned to some school:  
        sum over s of x[s, n, g] = Population_g[n]  (where g ∈ {White, NonWhite} and Population_g[n] is from neighborhoods_population.csv).
    -   **School capacity:** For each school s, total assigned students cannot exceed capacity:  
        sum over n and g of x[s, n, g] ≤ Capacity[s]  (from school_capacity.csv).
    -   **Racial balance:** For each school s, the percentage of white students assigned must be within 10 percentage points of the district-wide white percentage (i.e., between 50% and 70%):  
        Let TotalWhite = sum over all n of Population_White[n],  
        Let TotalStudents = sum over all n of (Population_White[n] + Population_NonWhite[n]),  
        DistrictWhitePct = TotalWhite / TotalStudents = 0.6.  
        For each school s:  
            0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7  
        (If denominator is zero, i.e., no students assigned, this is infeasible; but with full assignment and positive capacities, this will not occur.)
    -   **Non-negativity:** All x[s, n, g] ≥ 0.
[Abstract Model Plan END]