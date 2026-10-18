[Abstract Model Plan START]  
1. **Analyze Query:** The user wants to open a subset of ten candidate service centres (SC1–SC10) and assign each of fifteen customers (C1–C15) to exactly one opened centre, with the constraint that any opened centre can serve at most four customers. The goal is to minimise the sum of fixed opening costs for the chosen centres and the per‑customer service costs.  
2. **Identify Model Type:** This is a **fixed‑charge facility location problem** (a mixed‑integer linear program with binary opening variables and binary assignment variables).  
3. **Define Index Sets:**  
   - `SC` = {SC1, SC2, …, SC10} – the set of candidate service centres.  
   - `C`  = {C1, C2, …, C15} – the set of customers.  
4. **Define Decision Variables:**  
   - `y[j]` = 1 if service centre `j ∈ SC` is opened, 0 otherwise.  
     *Type:* `GRB.BINARY`.  
   - `x[i,j]` = 1 if customer `i ∈ C` is assigned to service centre `j ∈ SC`, 0 otherwise.  
     *Type:* `GRB.BINARY`.  
5. **Identify Parameters (from Schema):**  
   - **Fixed opening costs**: `fixed_cost[j]` comes from the column *Fixed Opening Cost* in `service_centers_fixed_costs.csv`.  
   - **Customer‑centre service costs**: `service_cost[i,j]` comes from the column *SCj* (e.g., *SC1*, *SC2*, …, *SC10*) in `expanded_customer_service_costs.csv`.  
   - **Capacity limit**: each centre can serve at most 4 customers (a constant `max_cap = 4`).  
6. **Formulate Objective:**  
   Minimise  
   \[
   \sum_{j \in SC} \text{fixed_cost}[j] \cdot y[j]
   \;+\;
   \sum_{i \in C}\sum_{j \in SC} \text{service_cost}[i,j] \cdot x[i,j].
   \]  
7. **Formulate Constraints:**  
   - **Assignment (each customer served once):**  
     \[
     \sum_{j \in SC} x[i,j] = 1 \quad \forall\, i \in C.
     \]  
   - **Linking (assignment only to opened centres):**  
     \[
     x[i,j] \le y[j] \quad \forall\, i \in C,\; j \in SC.
     \]  
   - **Capacity (max 4 customers per centre):**  
     \[
     \sum_{i \in C} x[i,j] \le 4 \quad \forall\, j \in SC.
     \]  
   - **Variable domains:**  
     \[
     y[j] \in \{0,1\},\; x[i,j] \in \{0,1\}.
     \]  

[Abstract Model Plan END]