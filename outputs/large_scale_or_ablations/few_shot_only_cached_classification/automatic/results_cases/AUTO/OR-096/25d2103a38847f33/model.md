[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from 31 neighborhoods to two schools, minimizing total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide 60% white / 40% nonwhite ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01–N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small enough that continuous is acceptable for planning).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n).
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
6.  **Formulate Objective:** Minimize the total travel distance for all students:  
        sum over all schools s, neighborhoods n, and groups g of (distance[s, n] * x[s, n, g])
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g,  
        sum over schools s of x[s, n, g] = Population_g[n]  
        (i.e., all white and nonwhite students from each neighborhood must be assigned to some school)
    -   **School capacity:** For each school s,  
        sum over all neighborhoods n and groups g of x[s, n, g] ≤ Capacity[s]
    -   **Racial balance:** For each school s,  
        Let total_white[s] = sum over n of x[s, n, White]  
        Let total_students[s] = sum over n and g of x[s, n, g]  
        Enforce:  
        0.5 ≤ total_white[s] / total_students[s] ≤ 0.7  
        (i.e., white percentage between 50% and 70%)
    -   **Non-negativity:** All x[s, n, g] ≥ 0
[Abstract Model Plan END]