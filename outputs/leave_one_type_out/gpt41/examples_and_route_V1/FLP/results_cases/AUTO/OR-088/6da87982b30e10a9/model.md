[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 candidate plants (F1–F15) to open and how much each open plant should supply to each of 15 customers (C1–C15), so as to minimize the total cost (fixed plant opening costs plus variable transportation costs), while meeting all customer demands and not exceeding plant capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge transportation) problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants: \( i \in \{\text{F1}, \ldots, \text{F15}\} \)
    - Customers: \( j \in \{\text{C1}, \ldots, \text{C15}\} \)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Amount shipped from plant \(i\) to customer \(j\). Type: GRB.CONTINUOUS (non-negative real).
    -   `y[i]` = 1 if plant \(i\) is opened (built), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each plant: from column `'fixed_cost'` in cost.csv.
    -   Plant capacity: from column `'capacity'` in cost.csv.
    -   Per-unit transport cost from plant \(i\) to customer \(j\): from columns `'C1'` to `'C15'` in cost.csv.
    -   Customer demand: from column `'demand'` in demand.csv, joined on customer label.
6.  **Formulate Objective:** Minimize total cost, which is the sum of:
    - Fixed opening costs for all opened plants: \(\sum_{i} \text{fixed\_cost}[i] \cdot y[i]\)
    - Plus total transportation costs: \(\sum_{i} \sum_{j} \text{transport\_cost}[i,j] \cdot x[i,j]\)
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer \(j\), the total amount received from all plants must equal their demand: \(\sum_{i} x[i,j] = \text{demand}[j]\)
    -   Constraint 2 (Plant Capacity): For each plant \(i\), the total amount shipped from that plant to all customers cannot exceed its capacity if opened: \(\sum_{j} x[i,j] \leq \text{capacity}[i] \cdot y[i]\)
    -   Constraint 3 (Linking): If a plant is not opened (\(y[i]=0\)), it cannot ship any product (\(x[i,j]=0\) for all \(j\)); this is enforced by the capacity constraint above.
    -   Constraint 4 (Variable Domains): \(x[i,j] \geq 0\) (continuous, non-negative); \(y[i] \in \{0,1\}\) (binary).
[Abstract Model Plan END]