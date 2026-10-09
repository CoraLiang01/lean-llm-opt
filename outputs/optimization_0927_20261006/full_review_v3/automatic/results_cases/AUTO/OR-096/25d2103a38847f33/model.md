[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity and each school's white-student percentage is within ±10 percentage points of the district's 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv and distance.csv ('School')
    - Neighborhoods (N): from neighborhoods_population.csv and distance.csv (columns 'N01'...'N31')
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    - School capacities: school_capacity.csv ('Capacity' for each 'School')
    - Neighborhood populations: neighborhoods_population.csv ('Population_White', 'Population_NonWhite' for each 'Neighborhood')
    - Distances: distance.csv (distance from each 'School' to each 'Neighborhood')
    - District-wide total white and nonwhite populations: sum over all neighborhoods
    - Racial balance target: 60% white (district ratio), with ±10% tolerance (i.e., each school must be 50%–70% white)
6.  **Formulate Objective:** Minimize the total travel distance for all students: sum over all schools, neighborhoods, and groups of (distance from school to neighborhood) × (number of students assigned).
7.  **Formulate Constraints:**
    - Assignment completeness: For each neighborhood n and group g, the sum over schools s of x[s, n, g] equals the total population of group g in neighborhood n (i.e., all students must be assigned to a school).
    - School capacity: For each school s, the sum over all neighborhoods n and both groups g of x[s, n, g] ≤ school s's capacity.
    - Racial balance: For each school s, the percentage of white students assigned must be between 50% and 70% of total students assigned to that school (i.e., 0.5 ≤ (sum over n of x[s, n, White]) / (sum over n and g of x[s, n, g]) ≤ 0.7), with appropriate handling if denominator is zero.
    - Nonnegativity: All x[s, n, g] ≥ 0.
[Abstract Model Plan END]