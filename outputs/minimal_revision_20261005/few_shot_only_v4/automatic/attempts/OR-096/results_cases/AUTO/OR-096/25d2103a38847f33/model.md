[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools: \( S = \{\text{I}, \text{II}\} \) (from school_capacity.csv)
    - Neighborhoods: \( N = \{\text{N01}, \ldots, \text{N31}\} \) (from neighborhoods_population.csv)
    - Student groups: \( G = \{\text{White}, \text{NonWhite}\} \) (from neighborhoods_population.csv)
4.  **Define Decision Variables:**
    -   \( x_{s,n,g} \) = Number of students of group \( g \) from neighborhood \( n \) assigned to school \( s \). Type: GRB.CONTINUOUS (nonnegative, can be restricted to integer if required, but not specified in query).
5.  **Identify Parameters (from Schema):**
    -   Student populations: 'Population_White', 'Population_NonWhite' from neighborhoods_population.csv (for each neighborhood).
    -   School capacities: 'Capacity' from school_capacity.csv (for each school).
    -   Distances: 'distance.csv' gives miles from each school to each neighborhood (for each school-neighborhood pair).
    -   District-wide white and nonwhite totals: sum over all neighborhoods.
6.  **Formulate Objective:** Minimize the total travel distance for all students:
        - Objective: Minimize sum over all schools \( s \), neighborhoods \( n \), and groups \( g \) of \( x_{s,n,g} \times \text{distance}_{s,n} \).
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood \( n \) and group \( g \), all students must be assigned to some school:
            \(\sum_{s} x_{s,n,g} = \text{Population}_{n,g}\) for all \( n \), \( g \).
    -   **School capacity:** For each school \( s \), total assigned students cannot exceed capacity:
            \(\sum_{n,g} x_{s,n,g} \leq \text{Capacity}_s\) for all \( s \).
    -   **Racial balance:** For each school \( s \), the percentage of white students assigned must be between 50% and 70%:
            \[
            0.5 \leq \frac{\sum_{n} x_{s,n,\text{White}}}{\sum_{n,g} x_{s,n,g}} \leq 0.7
            \]
            (If denominator is zero, school is empty; but with full assignment and positive capacities, this should not occur.)
    -   **Nonnegativity:** \( x_{s,n,g} \geq 0 \) for all \( s, n, g \).
[Abstract Model Plan END]