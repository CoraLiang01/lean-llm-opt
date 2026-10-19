[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign white and non‑white elementary‑school pupils from 31 neighborhoods to two schools so that each school’s capacity is not exceeded, each neighborhood’s pupils are fully assigned, each school’s white‑student proportion stays within ±10 percentage points of the district’s 60 % white target, and the total travel distance (in miles) is minimized.

2.  **Identify Model Type:** This is a **Linear Programming (LP)** problem with continuous, non‑negative decision variables.

3.  **Define Index Sets:**
    - **S** – set of schools: `{I, II}`  
    - **N** – set of neighborhoods: `{N01, N02, …, N31}`  
    - **R** – set of racial groups: `{White, NonWhite}`  

4.  **Define Decision Variables:**
    - `x[s,n,r]` = number of pupils of race `r` from neighborhood `n` assigned to school `s`.  
      Type: `GRB.CONTINUOUS`, domain `x[s,n,r] ≥ 0`.

5.  **Identify Parameters (from Schema):**
    - `Capacity[s]` – school capacity from `school_capacity.csv`.  
    - `PopWhite[n]`, `PopNonWhite[n]` – neighborhood white and non‑white populations from `neighborhoods_population.csv`.  
    - `Dist[s,n]` – distance (miles) from school `s` to neighborhood `n` from `distance.csv`.  
    - District white‑student target: `p_white = 0.60`.  
    - Allowed deviation: `δ = 0.10` → acceptable white proportion bounds: `p_min = p_white - δ = 0.50`, `p_max = p_white + δ = 0.70`.

6.  **Formulate Objective:**  
    Minimize total travel distance  
    \[
    \min \sum_{s \in S}\;\sum_{n \in N}\;\Dist[s,n]\;\bigl(x[s,n,White] + x[s,n,NonWhite]\bigr).
    \]

7.  **Formulate Constraints:**
    - **School capacity:**  
      \[
      \forall s \in S:\quad \sum_{n \in N}\;\bigl(x[s,n,White] + x[s,n,NonWhite]\bigr) \;\le\; Capacity[s].
      \]
    - **Neighborhood population balance:**  
      \[
      \forall n \in N:\quad
      \sum_{s \in S} x[s,n,White] \;=\; PopWhite[n],\qquad
      \sum_{s \in S} x[s,n,NonWhite] \;=\; PopNonWhite[n].
      \]
    - **Racial balance at each school:**  
      \[
      \forall s \in S:\quad
      p_{\min}\;\sum_{n \in N}\!\bigl(x[s,n,White] + x[s,n,NonWhite]\bigr)
      \;\le\;
      \sum_{n \in N} x[s,n,White]
      \;\le\;
      p_{\max}\;\sum_{n \in N}\!\bigl(x[s,n,White] + x[s,n,NonWhite]\bigr).
      \]
    - **Non‑negativity:**  
      \[
      \forall s \in S,\; \forall n \in N,\; \forall r \in R:\quad x[s,n,r] \;\ge\; 0.
      \]
[Abstract Model Plan END]