[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary school students from 31 neighborhoods to two schools, minimizing total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from `school_capacity.csv` (Schools I and II; all rows used).
    - Neighborhoods (N): from `neighborhoods_population.csv` (N01–N31; all rows used).
    - Student Groups (G): {White, NonWhite} (explicitly defined in the query and population columns).
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but not specified).
5.  **Identify Parameters (from Schema):**
    -   School capacities: `school_capacity.csv` column 'Capacity' for each school.
    -   Neighborhood populations: `neighborhoods_population.csv` columns 'Population_White' and 'Population_NonWhite' for each neighborhood.
    -   Distances: `distance.csv` entry for each (school, neighborhood) pair.
    -   District-wide white and nonwhite totals: sum over all neighborhoods.
6.  **Formulate Objective:** Minimize the total travel distance for all students:  
        sum over all schools s, neighborhoods n, and groups g of  
        (`distance[s, n]` * `x[s, n, g]`)
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood n and group g, all students must be assigned to some school:  
        sum over s of `x[s, n, g]` = `Population_g[n]` (where g ∈ {White, NonWhite}).
    -   **School Capacity:** For each school s, total assigned students cannot exceed capacity:  
        sum over n and g of `x[s, n, g]` ≤ `Capacity[s]`.
    -   **Racial Balance:** For each school s, the percentage of white students assigned must be within 50%–70%:  
        0.5 ≤ (sum over n of `x[s, n, White]`) / (sum over n and g of `x[s, n, g]`) ≤ 0.7,  
        with the denominator > 0 (since all students are assigned).
    -   **Nonnegativity:** All `x[s, n, g]` ≥ 0.
[Abstract Model Plan END]