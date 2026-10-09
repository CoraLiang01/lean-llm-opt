[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school’s enrollment does not exceed its capacity, each neighborhood’s students are fully assigned, and each school’s white-student percentage is within 10 percentage points of the district’s 60% white / 40% nonwhite ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) or Mixed-Integer Programming (MIP) transportation/assignment problem with side constraints.
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv`)
    - Neighborhoods (from `neighborhoods_population.csv`)
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.INTEGER (since students are indivisible).
5.  **Identify Parameters (from Schema):**
    -   Distance from each school to each neighborhood: from `distance.csv` (fields: School, N01–N31).
    -   School capacities: from `school_capacity.csv` (fields: School, Capacity).
    -   Neighborhood populations by group: from `neighborhoods_population.csv` (fields: Neighborhood, Population_White, Population_NonWhite).
    -   District-wide white and nonwhite student totals: sum over all neighborhoods.
    -   Racial balance target: 60% white, 40% nonwhite; allowable deviation: ±10 percentage points.
6.  **Formulate Objective:** Minimize the total travel distance for all students, i.e., sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned).
7.  **Formulate Constraints:**
    -   Assignment completeness: For each neighborhood and group, the sum of students assigned to all schools equals the group’s population in that neighborhood.
    -   School capacity: For each school, the total number of assigned students (all neighborhoods, both groups) does not exceed the school’s capacity.
    -   Racial balance: For each school, the percentage of white students among all assigned students must be between 50% and 70% (i.e., within ±10 percentage points of the district’s 60% white ratio).
    -   Nonnegativity and integrality: All assignment variables are integer and ≥ 0.
[Abstract Model Plan END]