[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from 31 neighborhoods to two schools, minimizing total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools: \( S = \{\text{I}, \text{II}\} \) (from school_capacity.csv)
    - Neighborhoods: \( N = \{\text{N01}, \ldots, \text{N31}\} \) (from neighborhoods_population.csv)
    - Student Groups: \( G = \{\text{White}, \text{NonWhite}\} \) (implied by population columns)
4.  **Define Decision Variables:**
    -   \( x_{s,n,g} \) = Number of students of group \( g \) from neighborhood \( n \) assigned to school \( s \). Type: GRB.CONTINUOUS (nonnegative, can be restricted to integer if required, but not specified in query).
5.  **Identify Parameters (from Schema):**
    -   Distance: \( d_{s,n} \) from distance.csv, columns "School" and each neighborhood.
    -   School capacity: \( \text{Capacity}_s \) from school_capacity.csv.
    -   Neighborhood populations: \( \text{Population\_White}_n \), \( \text{Population\_NonWhite}_n \) from neighborhoods_population.csv.
    -   District-wide white and nonwhite totals: sum over all neighborhoods.
6.  **Formulate Objective:** Minimize total student-miles traveled:  
        \( \min \sum_{s \in S} \sum_{n \in N} \sum_{g \in G} d_{s,n} \cdot x_{s,n,g} \)
7.  **Formulate Constraints:**
    -   **Assignment Completeness:** For each neighborhood \( n \) and group \( g \), all students must be assigned to a school:  
        \( \sum_{s \in S} x_{s,n,g} = \text{Population}_{g,n} \) for all \( n \in N, g \in G \)
    -   **School Capacity:** For each school \( s \), total assigned students cannot exceed capacity:  
        \( \sum_{n \in N} \sum_{g \in G} x_{s,n,g} \leq \text{Capacity}_s \) for all \( s \in S \)
    -   **Racial Balance:** For each school \( s \), the percentage of white students must be between 50% and 70%:  
        Let \( W_s = \sum_{n \in N} x_{s,n,\text{White}} \), \( T_s = \sum_{n \in N} \sum_{g \in G} x_{s,n,g} \).  
        Enforce:  
        \( 0.5 \cdot T_s \leq W_s \leq 0.7 \cdot T_s \) for all \( s \in S \)
    -   **Nonnegativity:** \( x_{s,n,g} \geq 0 \) for all \( s, n, g \)
[Abstract Model Plan END]