[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to minimize the total system cost of ElectroTech Manufacturing by deciding which of 15 potential factory sites to build and how to ship products from those sites to 8 distribution centers, while meeting fixed demand at each center.  
2.  **Identify Model Type:** This is a **Capacitated Facility Location Problem (CFLP)**, a mixed‑integer linear programming (MILP) model with fixed‑charge (opening) costs and variable shipping costs.  
3.  **Define Index Sets:**  
    - **Facilities** \(F = \{A1, A2, \dots, A15\}\)  
    - **Destinations** \(D = \{B1, B2, \dots, B8\}\)  
4.  **Define Decision Variables:**  
    - `x[i,j]` = quantity shipped from facility `i` to destination `j`.  
      Type: `GRB.CONTINUOUS`, \(x_{ij} \ge 0\).  
    - `y[i]` = 1 if facility `i` is built (opened), 0 otherwise.  
      Type: `GRB.BINARY`.  
5.  **Identify Parameters (from Schema):**  
    - **Fixed costs** `f[i]` come from column **FixedCost** in `facility_costs.csv`.  
    - **Facility capacities** `cap[i]` come from column **Capacity** in `facility_costs.csv`.  
    - **Shipping costs** `c[i,j]` come from columns **B1**–**B8** in `shipping_costs.csv`.  
    - **Demand** `d[j]` comes from column **Demand** in `demand_requirements.csv`.  
6.  **Formulate Objective:**  
    Minimize total cost  
    \[
    \min \;\sum_{i \in F} f[i]\,y[i] \;+\; \sum_{i \in F}\sum_{j \in D} c[i,j]\,x[i,j].
    \]  
7.  **Formulate Constraints:**  
    - **Demand satisfaction** (for each destination \(j\)):  
      \[
      \sum_{i \in F} x[i,j] \;=\; d[j].
      \]  
    - **Capacity limits** (for each facility \(i\)):  
      \[
      \sum_{j \in D} x[i,j] \;\le\; cap[i] \, y[i].
      \]  
    - **Non‑negativity**: \(x[i,j] \ge 0\) for all \(i,j\).  
    - **Binary facility decision**: \(y[i] \in \{0,1\}\) for all \(i\).  
[Abstract Model Plan END]