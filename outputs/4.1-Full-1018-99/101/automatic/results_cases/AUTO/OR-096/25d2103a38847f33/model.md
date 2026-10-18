[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school’s enrollment does not exceed its capacity and each school’s white-student percentage is within ±10 percentage points of the district’s 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv`)
    - Neighborhoods (from `neighborhoods_population.csv`)
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be integer if required).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity` from `school_capacity.csv` (indexed by `School`)
    - Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (indexed by `Neighborhood`)
    - Distances: `distance.csv` provides miles from each `School` to each `Neighborhood` (fields: `School`, `N01`, ..., `N31`)
    - District-wide white and nonwhite totals: sum over all neighborhoods of `Population_White` and `Population_NonWhite`
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned).
7.  **Formulate Constraints:**
    - **Neighborhood assignment:** For each neighborhood and group, the sum of students assigned to all schools equals the group’s population in that neighborhood (i.e., all students must be assigned).
    - **School capacity:** For each school, the total number of assigned students (all neighborhoods, both groups) does not exceed the school’s capacity.
    - **Racial balance:** For each school, the percentage of white students assigned must be within 50% to 70% (i.e., within ±10 percentage points of the district’s 60% white ratio). Formally, for each school:  
      0.5 ≤ (total white students assigned to school) / (total students assigned to school) ≤ 0.7, unless the denominator is zero (infeasible).
    - **Nonnegativity:** All assignment variables must be ≥ 0.
[Abstract Model Plan END]