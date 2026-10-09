[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and nonwhite elementary students from each of 31 neighborhoods to two schools, minimizing total travel distance, while ensuring each school's enrollment does not exceed its capacity and each school's white-student percentage is within 10 percentage points of the district's 60% white ratio.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation/assignment problem with side constraints (capacity and racial balance).
3.  **Define Index Sets:** The primary indices are:
    - Schools (from `school_capacity.csv` and `distance.csv`)
    - Neighborhoods (from `neighborhoods_population.csv` and `distance.csv`)
    - Student groups: {White, NonWhite}
4.  **Define Decision Variables:**
    - `x[s, n, g]` = Number of students of group `g` (White or NonWhite) from neighborhood `n` assigned to school `s`. Type: GRB.CONTINUOUS (nonnegative, can be integer if required by context).
5.  **Identify Parameters (from Schema):**
    - School capacities: `Capacity` from `school_capacity.csv` (indexed by `School`)
    - Neighborhood populations: `Population_White`, `Population_NonWhite` from `neighborhoods_population.csv` (indexed by `Neighborhood`)
    - Distances: `distance.csv` columns (distance from each `School` to each `Neighborhood`)
    - District-wide white and nonwhite totals: sum of `Population_White` and `Population_NonWhite` across all neighborhoods
    - Racial balance target: 60% white (district ratio), with ±10% tolerance (i.e., each school must be 50–70% white)
6.  **Formulate Objective:** Minimize the total distance traveled by all students:  
    sum over all schools `s`, neighborhoods `n`, and groups `g` of `distance[s, n] * x[s, n, g]`
7.  **Formulate Constraints:**
    - **Neighborhood assignment:** For each neighborhood `n` and group `g`,  
      sum over schools `s` of `x[s, n, g]` = `Population_g[n]` (all students from each group in each neighborhood must be assigned to some school)
    - **School capacity:** For each school `s`,  
      sum over all neighborhoods `n` and groups `g` of `x[s, n, g]` ≤ `Capacity[s]`
    - **Racial balance:** For each school `s`,  
      let `W_s` = sum over `n` of `x[s, n, White]`,  
      let `T_s` = sum over `n` and `g` of `x[s, n, g]`,  
      enforce: 0.5 × `T_s` ≤ `W_s` ≤ 0.7 × `T_s` (white percentage between 50% and 70%)
    - **Nonnegativity:** All `x[s, n, g]` ≥ 0
[Abstract Model Plan END]