[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to develop a busing plan for a school district that assigns white and nonwhite elementary students from each neighborhood to two schools, minimizing total travel distance, while ensuring each school’s capacity is not exceeded and each school’s racial composition is within 10 percentage points of the district-wide 60% white/40% nonwhite ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) or Mixed-Integer Programming (MIP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv`)
    - Neighborhoods (from `neighborhoods_population.csv` and `distance.csv`)
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.INTEGER (since students are indivisible).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from `school_capacity.csv` column `Capacity` (indexed by `School`).
    -   Neighborhood populations: from `neighborhoods_population.csv` columns `Population_White` and `Population_NonWhite` (indexed by `Neighborhood` and group).
    -   Distances: from `distance.csv` (distance from each `School` to each `Neighborhood`).
    -   District-wide racial ratio: 60% white, 40% nonwhite (from query).
    -   Racial balance tolerance: ±10 percentage points (from query).
6.  **Formulate Objective:** Minimize the total distance traveled by all students, i.e., sum over all schools, neighborhoods, and groups of (number of students assigned) × (distance from school to neighborhood).
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood and group, the total number of students assigned to all schools equals the group’s population in that neighborhood.
    -   **School Capacity:** For each school, the total number of students assigned (all neighborhoods, both groups) does not exceed the school’s capacity.
    -   **Racial Balance:** For each school, the percentage of white students assigned must be within 10 percentage points of the district-wide ratio (i.e., between 50% and 70% white).
    -   **Non-negativity and Integrality:** All assignment variables must be non-negative integers.
[Abstract Model Plan END]