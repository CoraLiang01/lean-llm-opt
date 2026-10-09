[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from 31 neighborhoods to 2 schools, minimizing total travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide 60% white / 40% nonwhite ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from school_capacity.csv): S = {I, II}
    - Neighborhoods (from neighborhoods_population.csv): N = {N01, N02, ..., N31}
    - Student groups: G = {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (or GRB.INTEGER if students must be assigned as whole individuals).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n) pair.
    -   District-wide racial ratio: 60% white, 40% nonwhite (from query).
6.  **Formulate Objective:** Minimize the total travel distance for all students:
        sum over s, n, g of (distance[s, n] * x[s, n, g])
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood n and group g, all students must be assigned to some school:
            sum over s of x[s, n, g] == Population_g[n]   (for g in {White, NonWhite})
    -   **School Capacity:** For each school s, total assigned students cannot exceed capacity:
            sum over n, g of x[s, n, g] <= Capacity[s]
    -   **Racial Balance:** For each school s, the percentage of white students assigned must be within 10 percentage points of 60% (i.e., between 50% and 70%):
            0.5 <= (sum over n of x[s, n, White]) / (sum over n, g of x[s, n, g]) <= 0.7
        (If the denominator is zero, the constraint is trivially satisfied, but in practice, each school will have students assigned.)
    -   **Non-negativity and Integrality:** x[s, n, g] >= 0, and (optionally) integer.
[Abstract Model Plan END]