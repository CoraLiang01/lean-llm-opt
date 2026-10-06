[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from 31 neighborhoods to two schools, minimizing total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide 60% white / 40% nonwhite ratio (i.e., each school must be 50–70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01–N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small enough that continuous will suffice for planning).
5.  **Identify Parameters (from Schema):**
    -   School capacities: school_capacity.csv, column 'Capacity' for each 'School'
    -   Neighborhood populations: neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each 'Neighborhood'
    -   Distances: distance.csv, entry for each (School, Neighborhood) pair
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned).
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, the sum over schools s of x[s, n, g] = total number of students of group g in neighborhood n (from neighborhoods_population.csv). This ensures all students are assigned to a school.
    -   **School capacity:** For each school s, the sum over all neighborhoods n and both groups g of x[s, n, g] ≤ school s's capacity (from school_capacity.csv).
    -   **Racial balance:** For each school s, the percentage of white students assigned must be between 50% and 70% of total enrollment at that school. That is, for each school s:
        - sum over n of x[s, n, 'White'] / (sum over n and g of x[s, n, g]) ∈ [0.5, 0.7], provided the denominator is positive (since all students must be assigned, this will be the case).
        - This can be linearized as: 0.5 × (total assigned to s) ≤ (total white assigned to s) ≤ 0.7 × (total assigned to s)
    -   **Non-negativity:** All x[s, n, g] ≥ 0, and cannot exceed the population of group g in neighborhood n.
[Abstract Model Plan END]