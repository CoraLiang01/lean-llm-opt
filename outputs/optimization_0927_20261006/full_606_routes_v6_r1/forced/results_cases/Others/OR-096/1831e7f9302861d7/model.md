[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity and each school's white-student percentage is within 10 percentage points of the district-wide 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv`)
    - Neighborhoods (from `neighborhoods_population.csv`)
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity` from `school_capacity.csv` (indexed by `School`)
    - Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (indexed by `Neighborhood`)
    - Distances: `distance.csv` provides miles from each `School` to each `Neighborhood` (fields: `School`, `N01`, ..., `N31`)
    - District-wide white and nonwhite totals: sum over all neighborhoods of `Population_White` and `Population_NonWhite`
6.  **Formulate Objective:** Minimize the total student-miles traveled: sum over all schools, neighborhoods, and groups of `distance[s, n] * x[s, n, g]`.
7.  **Formulate Constraints:**
    - **Neighborhood assignment:** For each neighborhood `n` and group `g`, the sum over schools of `x[s, n, g]` equals the total number of students of group `g` in neighborhood `n` (i.e., all students are assigned to a school).
    - **School capacity:** For each school `s`, the sum over all neighborhoods and both groups of `x[s, n, g]` does not exceed `Capacity[s]`.
    - **Racial balance:** For each school `s`, the percentage of white students assigned to the school must be within 10 percentage points of the district-wide white percentage (i.e., between 50% and 70% white). Formally, for each school, the ratio of total white students assigned to total students assigned must be between 0.5 and 0.7.
    - **Nonnegativity:** All `x[s, n, g]` ≥ 0.
[Abstract Model Plan END]