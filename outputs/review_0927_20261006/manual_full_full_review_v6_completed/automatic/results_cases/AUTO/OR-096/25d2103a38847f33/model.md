[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school’s enrollment does not exceed its capacity and each school’s white-student percentage is within 10 percentage points of the district-wide 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv` and `distance.csv`)
    - Neighborhoods (from `neighborhoods_population.csv` and `distance.csv`)
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity[s]` from `school_capacity.csv`
    - Neighborhood populations: `Population_White[n]`, `Population_NonWhite[n]` from `neighborhoods_population.csv`
    - Distances: `Distance[s, n]` from `distance.csv` (distance from school `s` to neighborhood `n`)
    - District-wide white student ratio: 60% (from query)
    - Racial balance tolerance: ±10 percentage points (from query)
6.  **Formulate Objective:** Minimize the total distance traveled by all students:  
    `minimize sum over s, n, g of (Distance[s, n] * x[s, n, g])`
7.  **Formulate Constraints:**
    - **Neighborhood population assignment:** For each neighborhood `n` and group `g`,  
      `sum over s of x[s, n, g] = Population_g[n]`  
      (All students from each group in each neighborhood must be assigned to some school.)
    - **School capacity:** For each school `s`,  
      `sum over n, g of x[s, n, g] <= Capacity[s]`  
      (Total assigned students at each school cannot exceed its capacity.)
    - **Racial balance at each school:** For each school `s`,  
      Let `W_s = sum over n of x[s, n, White]` and `T_s = sum over n, g of x[s, n, g]`  
      Enforce:  
      `0.50 * T_s <= W_s <= 0.70 * T_s`  
      (White percentage at each school must be between 50% and 70%, i.e., within ±10% of the district’s 60%.)
    - **Nonnegativity:** All `x[s, n, g] >= 0`
[Abstract Model Plan END]