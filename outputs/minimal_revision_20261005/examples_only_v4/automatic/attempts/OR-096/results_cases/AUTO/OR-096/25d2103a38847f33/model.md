[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary school students from each of 31 neighborhoods to two schools, so that (a) all students are assigned, (b) no school exceeds its capacity, (c) each school’s white-student percentage is within 10 percentage points of the district-wide 60% white/40% nonwhite ratio, and (d) the total student travel distance is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools (S): from school_capacity.csv (I, II)
    - Neighborhoods (N): from neighborhoods_population.csv (N01, ..., N31)
    - Student groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but population numbers are small so integer solutions are likely).
5.  **Identify Parameters (from Schema):**
    -   School capacities: from school_capacity.csv, column 'Capacity' for each school s.
    -   Neighborhood populations: from neighborhoods_population.csv, columns 'Population_White' and 'Population_NonWhite' for each neighborhood n.
    -   Distances: from distance.csv, entry for each (school s, neighborhood n) pair.
    -   District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
6.  **Formulate Objective:** Minimize the total travel distance for all students, i.e., sum over all schools, neighborhoods, and groups of (distance from school s to neighborhood n) × (number of students of group g assigned from n to s):  
        Minimize ∑_{s ∈ S} ∑_{n ∈ N} ∑_{g ∈ G} distance[s, n] × x[s, n, g]
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood n and group g, all students must be assigned to some school:  
            ∑_{s ∈ S} x[s, n, g] = Population_g[n]  
        (where Population_g[n] is 'Population_White' or 'Population_NonWhite' for group g in neighborhood n)
    -   **School capacity:** For each school s, the total number of assigned students cannot exceed its capacity:  
            ∑_{n ∈ N} ∑_{g ∈ G} x[s, n, g] ≤ Capacity[s]
    -   **Racial balance:** For each school s, the percentage of white students assigned must be within 10 percentage points of the district-wide ratio (i.e., between 50% and 70% white):  
            0.5 ≤ (total white students assigned to s) / (total students assigned to s) ≤ 0.7  
        That is,  
            0.5 × (total students assigned to s) ≤ (total white students assigned to s) ≤ 0.7 × (total students assigned to s)  
        where  
            total white students assigned to s = ∑_{n ∈ N} x[s, n, White]  
            total students assigned to s = ∑_{n ∈ N} ∑_{g ∈ G} x[s, n, g}
    -   **Nonnegativity:** All x[s, n, g] ≥ 0; (integer if required by context).
[Abstract Model Plan END]