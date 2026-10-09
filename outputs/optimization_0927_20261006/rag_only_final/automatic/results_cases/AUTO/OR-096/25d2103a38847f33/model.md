[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring that (a) no school exceeds its capacity, (b) all students are assigned, and (c) each school’s white-student percentage is within 10 percentage points of the district-wide ratio (60% white, 40% nonwhite).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    -   Schools (S): from `school_capacity.csv` and `distance.csv` (School column)
    -   Neighborhoods (N): from `neighborhoods_population.csv` and `distance.csv` (Neighborhood columns N01–N31)
    -   Student Groups (G): {White, NonWhite}
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group g (White or NonWhite) from neighborhood n assigned to school s. Type: GRB.CONTINUOUS (or GRB.INTEGER if students must be assigned as whole individuals).
5.  **Identify Parameters (from Schema):**
    -   `Capacity[s]`: School capacity from `school_capacity.csv` (Capacity column).
    -   `Pop[n, g]`: Number of students of group g in neighborhood n from `neighborhoods_population.csv` (Population_White, Population_NonWhite).
    -   `Dist[s, n]`: Distance from school s to neighborhood n from `distance.csv`.
    -   District-wide white and nonwhite totals: sum over all n of `Pop[n, White]` and `Pop[n, NonWhite]` (computed from `neighborhoods_population.csv`).
6.  **Formulate Objective:** Minimize the total travel distance for all students:  
    sum over s, n, g of `Dist[s, n] * x[s, n, g]`.
7.  **Formulate Constraints:**
    -   Assignment Completeness: For each neighborhood n and group g,  
        sum over s of `x[s, n, g]` = `Pop[n, g]` (all students assigned).
    -   School Capacity: For each school s,  
        sum over n, g of `x[s, n, g]` ≤ `Capacity[s]`.
    -   Racial Balance: For each school s,  
        Let `TotalWhite[s]` = sum over n of `x[s, n, White]`,  
        Let `TotalAssigned[s]` = sum over n, g of `x[s, n, g]`,  
        Enforce:  
        0.50 × `TotalAssigned[s]` ≤ `TotalWhite[s]` ≤ 0.70 × `TotalAssigned[s]`  
        (since district ratio is 60% white, ±10 percentage points).
    -   Nonnegativity and Integrality: All `x[s, n, g]` ≥ 0 and integer (if required).
[Abstract Model Plan END]