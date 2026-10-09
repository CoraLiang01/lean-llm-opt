[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, so as to minimize total student travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within ±10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools: \( S = \{\text{I}, \text{II}\} \) (from school_capacity.csv, all rows)
    - Neighborhoods: \( N = \{\text{N01}, \ldots, \text{N31}\} \) (from neighborhoods_population.csv, all rows)
    - Student groups: \( G = \{\text{White}, \text{NonWhite}\} \) (from neighborhoods_population.csv columns)
4.  **Define Decision Variables:**
    -   `x[s, n, g]` = Number of students of group \( g \) from neighborhood \( n \) assigned to school \( s \). Type: GRB.CONTINUOUS (nonnegative, can be integer if required, but not specified in query).
5.  **Identify Parameters (from Schema):**
    -   Student populations: `Population_White`, `Population_NonWhite` (from neighborhoods_population.csv, for each neighborhood)
    -   School capacities: `Capacity` (from school_capacity.csv, for each school)
    -   Distances: `distance[s][n]` (from distance.csv, for each school-neighborhood pair)
    -   District-wide white and nonwhite totals: sum over all neighborhoods of `Population_White` and `Population_NonWhite`
    -   Racial balance target: 60% white (±10 percentage points, i.e., 50%–70%)
6.  **Formulate Objective:** Minimize the total student-miles traveled:
    -   Minimize \( \sum_{s \in S} \sum_{n \in N} \sum_{g \in G} x[s, n, g] \times \text{distance}[s][n] \)
7.  **Formulate Constraints:**
    -   **Assignment completeness:** For each neighborhood \( n \) and group \( g \), all students must be assigned:
        -   \( \sum_{s \in S} x[s, n, g] = \text{Population}_{g}[n] \) for all \( n \in N, g \in G \)
    -   **School capacity:** For each school \( s \), total assigned students cannot exceed capacity:
        -   \( \sum_{n \in N} \sum_{g \in G} x[s, n, g] \leq \text{Capacity}[s] \) for all \( s \in S \)
    -   **Racial balance:** For each school \( s \), the percentage of white students assigned must be between 50% and 70%:
        -   \( 0.5 \leq \frac{\sum_{n \in N} x[s, n, \text{White}]}{\sum_{n \in N} \sum_{g \in G} x[s, n, g]} \leq 0.7 \) for all \( s \in S \)
        -   (This can be linearized by cross-multiplying denominators if needed.)
    -   **Nonnegativity:** \( x[s, n, g] \geq 0 \) for all \( s \in S, n \in N, g \in G \)
[Abstract Model Plan END]