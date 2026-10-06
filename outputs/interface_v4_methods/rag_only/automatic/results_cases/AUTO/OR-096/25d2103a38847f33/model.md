[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from 31 neighborhoods to 2 schools, minimizing total travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide ratio (60% white, 40% nonwhite).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) or Mixed-Integer Programming (MIP) transportation/assignment problem with side constraints (racial balance).
3.  **Define Index Sets:** The primary indices are:
    -   Schools (S): from `school_capacity.csv` and `distance.csv` (2 schools: I, II)
    -   Neighborhoods (N): from `neighborhoods_population.csv` and `distance.csv` (31 neighborhoods: N01–N31)
    -   Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.INTEGER (since students are indivisible).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from `school_capacity.csv` ('Capacity' column, keyed by 'School')
    -   Neighborhood populations: from `neighborhoods_population.csv` ('Population_White', 'Population_NonWhite', keyed by 'Neighborhood')
    -   Distances: from `distance.csv` (distance from each school to each neighborhood, columns N01–N31, keyed by 'School')
    -   District-wide racial ratio: 60% white, 40% nonwhite (from query, not schema)
6.  **Formulate Objective:** Minimize the total distance traveled by all students, i.e.,  
    sum over all schools s, neighborhoods n, and groups g of:  
    `distance[s, n] * x[s, n, g]`
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood n and group g,  
        sum over all schools s of `x[s, n, g]` = total number of students of group g in neighborhood n (from `neighborhoods_population.csv`).
    -   **School Capacity:** For each school s,  
        sum over all neighborhoods n and groups g of `x[s, n, g]` ≤ school capacity (from `school_capacity.csv`).
    -   **Racial Balance:** For each school s,  
        the percentage of white students assigned to school s must be between 50% and 70% (i.e., within ±10 percentage points of the district ratio of 60% white).  
        Formally:  
        0.5 ≤ (total white students assigned to s) / (total students assigned to s) ≤ 0.7  
        (Handle denominator = 0 case if needed.)
    -   **Non-negativity and Integrality:** All `x[s, n, g]` ≥ 0 and integer.
[Abstract Model Plan END]