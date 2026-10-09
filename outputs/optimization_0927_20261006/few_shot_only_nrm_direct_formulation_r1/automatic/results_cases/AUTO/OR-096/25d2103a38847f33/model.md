[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity and each school's white-student percentage is within 10 percentage points of the district's 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from `school_capacity.csv` (all rows, i.e., School I and II)
    - Neighborhoods (N): from `neighborhoods_population.csv` (all 31 rows)
    - Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be fractional if partial assignments are allowed; otherwise, GRB.INTEGER if only whole students are allowed).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity` from `school_capacity.csv` (by School)
    - Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (by Neighborhood)
    - Distances: `distance.csv` (distance from each School to each Neighborhood)
    - District-wide total white and nonwhite student counts: sum over all neighborhoods
    - District white ratio: 60% (from query)
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned).
7.  **Formulate Constraints:**
    - **Neighborhood Assignment:** For each neighborhood n and group g, the sum over schools s of x[s, n, g] equals the total number of students of group g in neighborhood n (from `neighborhoods_population.csv`).
    - **School Capacity:** For each school s, the total number of students assigned (sum over all neighborhoods and both groups) does not exceed the school's capacity (from `school_capacity.csv`).
    - **Racial Balance:** For each school s, the percentage of white students assigned must be within 10 percentage points of the district white ratio (i.e., between 50% and 70% white). That is, for each school s:  
      0.5 ≤ (total white students assigned to s) / (total students assigned to s) ≤ 0.7, with denominators > 0.
    - **Nonnegativity:** All x[s, n, g] ≥ 0.
[Abstract Model Plan END]