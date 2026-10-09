[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity, all students are assigned, and each school's white-student percentage is within ±10 percentage points of the district's 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv` and `distance.csv`): set S
    - Neighborhoods (from `neighborhoods_population.csv` and `distance.csv`): set N
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    -   School capacities: `Capacity` from `school_capacity.csv` (indexed by School s)
    -   Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (indexed by Neighborhood n)
    -   Distances: `distance.csv` columns (distance from school s to neighborhood n)
    -   District-wide white and nonwhite totals: sum over all neighborhoods of `Population_White` and `Population_NonWhite`
    -   Racial balance target: 60% white (±10 percentage points)
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools s, neighborhoods n, and groups g of (distance from s to n) × x[s, n, g].
7.  **Formulate Constraints:**
    -   Assignment completeness: For each neighborhood n and group g, sum over schools s of x[s, n, g] = total population of group g in neighborhood n (from `neighborhoods_population.csv`).
    -   School capacity: For each school s, sum over all neighborhoods n and both groups g of x[s, n, g] ≤ Capacity of school s (from `school_capacity.csv`).
    -   Racial balance: For each school s, the proportion of white students assigned to s must be between 50% and 70% of total students assigned to s (i.e., 0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7), with appropriate handling for zero denominator.
    -   Nonnegativity: All x[s, n, g] ≥ 0.
[Abstract Model Plan END]