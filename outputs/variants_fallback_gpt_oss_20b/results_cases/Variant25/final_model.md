**

The model is a classic assignment problem with the following components:

- **Sets**  
  - \(W\): workers with `on_leave = 0`.  
  - \(P\): projects.

- **Parameters**  
  - `skill_lvl[w]`: integer skill level of worker \(w\) (Junior = 1, Intermediate = 2, Senior = 3, Expert = 4).  
  - `req_lvl[p]`: required skill level for project \(p\).  
  - `c[w,p]`: assignment cost in USD cents (0 if the cell is blank).  
  - `allowed[w,p]`: 1 if worker \(w\) can be assigned to project \(p\) (cost defined **and** skill sufficient), 0 otherwise.

- **Decision Variables**  
  - \(x_{w,p} \in \{0,1\}\): 1 if worker \(w\) is assigned to project \(p\).

- **Objective**  
  \[
  \min \sum_{w\in W}\sum_{p\in P} c[w,p]\,x_{w,p}
  \]

- **Constraints**  
  1. Each project receives exactly one worker:  
     \[
     \sum_{w\in W} x_{w,p} = 1 \quad \forall p\in P
     \]
  2. Each worker takes at most one project:  
     \[
     \sum_{p\in P} x_{w,p} \le 1 \quad \forall w\in W
     \]
  3. Feasibility: assignments only allowed where `allowed[w,p] = 1`:  
     \[
     x_{w,p} \le \text{allowed}[w,p] \quad \forall w\in W,\;p\in P
     \]

**CODE:**