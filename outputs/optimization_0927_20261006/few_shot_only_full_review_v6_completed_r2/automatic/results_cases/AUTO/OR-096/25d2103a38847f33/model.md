[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity and each school's white-student percentage is within 10 percentage points of the district's 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv`): \( S \)
    - Neighborhoods (from `neighborhoods_population.csv`): \( N \)
    - Student groups: {White, NonWhite} (\( G \))
4.  **Define Decision Variables:**
    - \( x_{s,n,g} \) = Number of students of group \( g \) (White or NonWhite) from neighborhood \( n \) assigned to school \( s \). Type: GRB.CONTINUOUS (nonnegative, can be fractional if partial assignments are allowed; otherwise, GRB.INTEGER if only whole students are allowed).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity` from `school_capacity.csv` (indexed by school \( s \))
    - Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (indexed by neighborhood \( n \))
    - Distances: `distance.csv` provides distance in miles from each school \( s \) to each neighborhood \( n \)
    - District-wide white and nonwhite totals: sum over all neighborhoods of `Population_White` and `Population_NonWhite`
6.  **Formulate Objective:** Minimize the total student-miles traveled: sum over all schools \( s \), neighborhoods \( n \), and groups \( g \) of \( x_{s,n,g} \times \text{distance}_{s,n} \).
7.  **Formulate Constraints:**
    - **Neighborhood assignment:** For each neighborhood \( n \) and group \( g \), the sum over schools \( s \) of \( x_{s,n,g} \) equals the total population of group \( g \) in neighborhood \( n \) (i.e., all students must be assigned to a school).
    - **School capacity:** For each school \( s \), the sum over all neighborhoods \( n \) and groups \( g \) of \( x_{s,n,g} \) does not exceed the school's capacity.
    - **Racial balance:** For each school \( s \), the percentage of white students assigned must be within 10 percentage points of the district's white percentage (i.e., between 50% and 70% white). This is: \( 0.5 \leq \frac{\text{total white at } s}{\text{total students at } s} \leq 0.7 \), where totals are sums over all neighborhoods.
    - **Nonnegativity:** All \( x_{s,n,g} \geq 0 \).
[Abstract Model Plan END]