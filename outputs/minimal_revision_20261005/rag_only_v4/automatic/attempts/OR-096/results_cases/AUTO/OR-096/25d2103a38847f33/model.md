[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from 31 neighborhoods to 2 schools, minimizing total travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide ratio (60% white, 40% nonwhite).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) or Mixed-Integer Programming (MIP) transportation/assignment problem with side constraints (racial balance).
3.  **Define Index Sets:** The primary indices are:
    -   Schools (from school_capacity.csv): S = {I, II}
    -   Neighborhoods (from neighborhoods_population.csv): N = {N01, N02, ..., N31}
    -   Student groups: G = {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.INTEGER (since students are indivisible).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n) pair.
    -   District-wide racial ratio: 60% white, 40% nonwhite (from query, not schema).
6.  **Formulate Objective:** Minimize the total distance traveled by all students, i.e., sum over all schools, neighborhoods, and groups of (distance from school s to neighborhood n) × (number of students of group g from n assigned to s).
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, the sum over schools s of x[s, n, g] equals the total number of students of group g in neighborhood n (from neighborhoods_population.csv).
    -   **School capacity:** For each school s, the sum over all neighborhoods n and both groups g of x[s, n, g] ≤ school capacity (from school_capacity.csv).
    -   **Racial balance:** For each school s, the percentage of white students assigned to s must be between 50% and 70% of total students assigned to s (i.e., within ±10 percentage points of the district ratio). Formally, for each school s:  
        0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7, provided the denominator is nonzero.
    -   **Non-negativity and integrality:** All x[s, n, g] ≥ 0 and integer.
[Abstract Model Plan END]