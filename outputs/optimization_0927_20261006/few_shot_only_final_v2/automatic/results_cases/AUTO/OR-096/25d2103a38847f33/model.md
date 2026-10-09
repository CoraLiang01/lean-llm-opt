[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity, all students are assigned, and each school's white-student percentage is within ±10 percentage points of the district's 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from `school_capacity.csv` (all rows, i.e., School I and II)
    - Neighborhoods (N): from `neighborhoods_population.csv` (all 31 neighborhoods)
    - Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    - School capacities: `school_capacity.csv` column 'Capacity' for each school s.
    - Neighborhood populations: `neighborhoods_population.csv` columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    - Distances: `distance.csv` entry for each school s and neighborhood n.
    - District-wide total white and nonwhite student counts: sum over all neighborhoods from `neighborhoods_population.csv`.
    - District white ratio: fixed at 60% (from query).
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools s, neighborhoods n, and groups g of (distance from s to n) × x[s, n, g].
7.  **Formulate Constraints:**
    - Assignment completeness: For each neighborhood n and group g, sum over schools s of x[s, n, g] = neighborhood population of group g (from `neighborhoods_population.csv`).
    - School capacity: For each school s, sum over all neighborhoods n and groups g of x[s, n, g] ≤ school capacity (from `school_capacity.csv`).
    - Racial balance: For each school s, the proportion of white students assigned must be between 50% and 70% (i.e., within ±10 percentage points of the 60% district ratio):  
      0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7, for each school s (with denominators > 0).
    - Nonnegativity: All x[s, n, g] ≥ 0.
[Abstract Model Plan END]