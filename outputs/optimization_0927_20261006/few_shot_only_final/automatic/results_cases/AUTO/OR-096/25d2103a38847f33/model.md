[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity and each school's white-student percentage is within ±10 percentage points of the district's 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from `school_capacity.csv` (all rows, i.e., School I and II)
    - Neighborhoods (N): from `neighborhoods_population.csv` (all rows, N01–N31)
    - Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity` from `school_capacity.csv` (by School)
    - Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (by Neighborhood)
    - Distances: `distance.csv` columns (distance from each School to each Neighborhood)
    - District-wide total white and nonwhite student counts: sum over all neighborhoods
    - Racial balance target: 60% white (±10 percentage points)
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned).
7.  **Formulate Constraints:**
    - Assignment completeness: For each neighborhood n and group g, the sum over schools s of `x[s, n, g]` equals the total number of students of group g in neighborhood n (from `neighborhoods_population.csv`).
    - School capacity: For each school s, the sum over all neighborhoods n and groups g of `x[s, n, g]` ≤ school s's capacity (from `school_capacity.csv`).
    - Racial balance: For each school s, the percentage of white students assigned to s must be between 50% and 70% of total students assigned to s (i.e., 60% ± 10%). That is, for each s:  
      0.5 ≤ (sum over n of `x[s, n, White]`) / (sum over n and g of `x[s, n, g]`) ≤ 0.7, with appropriate handling if denominator is nonzero.
    - Nonnegativity: All `x[s, n, g]` ≥ 0.
[Abstract Model Plan END]