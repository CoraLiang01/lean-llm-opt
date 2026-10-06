[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white / 40% nonwhite ratio (i.e., each school must be 50–70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from all rows in school_capacity.csv (I, II)
    - Neighborhoods (N): from all rows in neighborhoods_population.csv (N01–N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small enough that continuous is acceptable for planning).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' (per school s)
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' (per neighborhood n)
    -   Distances: from distance.csv, columns for each neighborhood n, per school s (distance[s, n])
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'
6.  **Formulate Objective:** Minimize the total travel distance for all students:  
        sum over all schools s, neighborhoods n, and groups g of (distance[s, n] * x[s, n, g])
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, all students must be assigned to some school:  
        sum over s of x[s, n, g] = Population_g[n]  (where g ∈ {White, NonWhite})
    -   **School capacity:** For each school s, total assigned students cannot exceed capacity:  
        sum over n and g of x[s, n, g] ≤ Capacity[s]
    -   **Racial balance:** For each school s, the percentage of white students assigned must be within 50%–70% of total assigned students:  
        0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7  
        (If the denominator is zero, the school is empty; in practice, this will not occur due to assignment completeness and capacity constraints.)
    -   **Non-negativity:** All x[s, n, g] ≥ 0
[Abstract Model Plan END]