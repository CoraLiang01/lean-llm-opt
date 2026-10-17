[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary school students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring: (a) each school's enrollment does not exceed its capacity, (b) all students are assigned, and (c) each school's white-student percentage is within 10 percentage points of the district-wide 60% white ratio (i.e., between 50% and 70% white).
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (racial balance and capacity).
3.  **Define Index Sets:** The primary indices are:
    - Schools: $s \in \{\text{I}, \text{II}\}$ (from school_capacity.csv)
    - Neighborhoods: $n \in \{\text{N01}, \ldots, \text{N31}\}$ (from neighborhoods_population.csv)
    - Student groups: $g \in \{\text{White}, \text{NonWhite}\}$
4.  **Define Decision Variables:**
    - $x_{s,n,g}$ = Number of students of group $g$ from neighborhood $n$ assigned to school $s$. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    - School capacities: school_capacity.csv, field 'Capacity' for each 'School' $s$.
    - Neighborhood populations: neighborhoods_population.csv, fields 'Population_White' and 'Population_NonWhite' for each 'Neighborhood' $n$.
    - Distances: distance.csv, field for each 'School' $s$ and 'Neighborhood' $n$ (distance in miles).
    - District-wide white and nonwhite totals: sum over all neighborhoods of 'Population_White' and 'Population_NonWhite'.
6.  **Formulate Objective:** Minimize the total student-miles traveled: $\sum_{s,n,g} x_{s,n,g} \times \text{distance}_{s,n}$
7.  **Formulate Constraints:**
    - **Assignment completeness:** For each neighborhood $n$ and group $g$, $\sum_{s} x_{s,n,g} = \text{Population}_{n,g}$ (all students assigned).
    - **School capacity:** For each school $s$, $\sum_{n,g} x_{s,n,g} \leq \text{Capacity}_s$ (do not exceed school capacity).
    - **Racial balance:** For each school $s$, the proportion of white students assigned must be between 50% and 70%:
        - $0.5 \leq \frac{\sum_{n} x_{s,n,\text{White}}}{\sum_{n,g} x_{s,n,g}} \leq 0.7$ (if denominator $>0$; if a school is empty, the ratio is undefined but this is not relevant here).
    - **Nonnegativity:** $x_{s,n,g} \geq 0$ for all $s, n, g$.
[Abstract Model Plan END]