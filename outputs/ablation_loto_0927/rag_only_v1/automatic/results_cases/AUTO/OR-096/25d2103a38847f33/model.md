[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from 31 neighborhoods to 2 schools, minimizing total travel distance, while ensuring each school’s enrollment does not exceed its capacity, each neighborhood’s students are fully assigned, and each school’s white-student percentage is within 10 percentage points of the district’s 60% white / 40% nonwhite ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance).
3.  **Define Index Sets:** The primary indices are:
    -   Schools (from school_capacity.csv): S = {I, II}
    -   Neighborhoods (from neighborhoods_population.csv): N = {N01, N02, ..., N31}
    -   Student groups: G = {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (can be integer if required, but not specified).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n).
    -   District-wide racial ratio: 60% white, 40% nonwhite (from query).
    -   Racial balance tolerance: ±10 percentage points (from query).
6.  **Formulate Objective:** Minimize the total distance traveled by all students, i.e.,  
    sum over all s, n, g of (distance[s, n] * x[s, n, g])
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood n and group g,  
        sum over s of x[s, n, g] = Population_g[n]  
        (All white and nonwhite students from each neighborhood must be assigned to a school.)
    -   **School Capacity:** For each school s,  
        sum over n and g of x[s, n, g] ≤ Capacity[s]  
        (Total assigned students to each school cannot exceed its capacity.)
    -   **Racial Balance:** For each school s,  
        Let total_white[s] = sum over n of x[s, n, White]  
        Let total_nonwhite[s] = sum over n of x[s, n, NonWhite]  
        Let total_students[s] = total_white[s] + total_nonwhite[s]  
        Enforce:  
        0.50 ≤ total_white[s] / total_students[s] ≤ 0.70  
        (White percentage in each school must be within 10 percentage points of 60%.)
    -   **Non-negativity:** All x[s, n, g] ≥ 0.
[Abstract Model Plan END]