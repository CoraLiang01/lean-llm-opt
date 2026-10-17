[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring: (a) each school's total assigned students do not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white at each school).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01, ..., N31)
    - Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but not specified in query).
5.  **Identify Parameters (from Schema):**
    - School capacities: school_capacity.csv, field 'Capacity' for each school s.
    - Neighborhood populations: neighborhoods_population.csv, fields 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    - Distances: distance.csv, field for each (school s, neighborhood n) pair.
    - District-wide white student ratio: computed as total white students divided by total students across all neighborhoods (from neighborhoods_population.csv).
    - Racial balance tolerance: ±10 percentage points from 60% (i.e., 50% ≤ white% ≤ 70% at each school).
6.  **Formulate Objective:** Minimize the total distance traveled by all students: sum over all schools s, neighborhoods n, and groups g of (distance[s, n] * x[s, n, g]).
7.  **Formulate Constraints:**
    - Assignment completeness: For each neighborhood n and group g, sum over schools s of x[s, n, g] = population of group g in neighborhood n (from neighborhoods_population.csv).
    - School capacity: For each school s, sum over all neighborhoods n and groups g of x[s, n, g] ≤ capacity of school s (from school_capacity.csv).
    - Racial balance at each school: For each school s, the proportion of white students assigned must be between 50% and 70%:
        - 0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7, for each school s. (If denominator is zero, school is empty and constraint is vacuously satisfied.)
    - Nonnegativity: All x[s, n, g] ≥ 0.
[Abstract Model Plan END]