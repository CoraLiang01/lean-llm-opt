[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity and each school's white-student percentage is within 10 percentage points of the district-wide 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv`)
    - Neighborhoods (from `neighborhoods_population.csv`)
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be fractional if partial assignments are allowed; otherwise, GRB.INTEGER if only whole students are allowed).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity` from `school_capacity.csv` (indexed by `School`)
    - Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (indexed by `Neighborhood`)
    - Distances: `distance.csv` provides miles from each `School` to each `Neighborhood`
    - District-wide white and nonwhite totals: sum over all neighborhoods of `Population_White` and `Population_NonWhite`
    - Racial balance target: 60% white (district ratio), with allowable deviation ±10 percentage points (i.e., each school must have white percentage between 50% and 70%)
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (number of students assigned) × (distance from school to neighborhood).
7.  **Formulate Constraints:**
    - **Neighborhood population assignment:** For each neighborhood and group, the sum of students assigned to all schools equals the total population of that group in the neighborhood (i.e., all students must be assigned to a school).
    - **School capacity:** For each school, the total number of students assigned (all neighborhoods, both groups) does not exceed the school's capacity.
    - **Racial balance:** For each school, the percentage of white students assigned must be between 50% and 70% of the total assigned to that school.
    - **Nonnegativity:** All assignment variables must be ≥ 0.
[Abstract Model Plan END]