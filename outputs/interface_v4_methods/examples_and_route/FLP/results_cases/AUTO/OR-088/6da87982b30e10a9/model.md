[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 candidate plants (F1–F15) to build and how to assign shipments from these plants to 15 customers (C1–C15) so that all customer demands are met, no plant exceeds its capacity, and the total cost (fixed plant opening costs plus per-unit transportation costs) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge transportation) problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants: \( i \in \{\text{F1}, \ldots, \text{F15}\} \)
    - Customers: \( j \in \{\text{C1}, \ldots, \text{C15}\} \)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Amount shipped from plant \(i\) to customer \(j\). Type: GRB.CONTINUOUS (non-negative real numbers).
    -   `y[i]` = 1 if plant \(i\) is built (opened), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each plant: from column 'fixed_cost' in cost.csv.
    -   Plant capacity: from column 'capacity' in cost.csv.
    -   Per-unit transport cost from plant \(i\) to customer \(j\): from columns 'C1'–'C15' in cost.csv.
    -   Customer demand: from column 'demand' in demand.csv.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    -   The fixed opening costs for all built plants: \(\sum_{i} \text{fixed\_cost}[i] \cdot y[i]\)
    -   The total transportation cost: \(\sum_{i} \sum_{j} \text{transport\_cost}[i,j] \cdot x[i,j]\)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each customer \(j\), the total amount received from all plants must meet their demand:
        -   \(\sum_{i} x[i,j] = \text{demand}[j]\) for all \(j\)
    -   **Plant Capacity:** For each plant \(i\), the total amount shipped from the plant cannot exceed its capacity if it is built:
        -   \(\sum_{j} x[i,j] \leq \text{capacity}[i] \cdot y[i]\) for all \(i\)
    -   **Linking Constraint:** If a plant is not built (\(y[i]=0\)), it cannot ship any product (\(x[i,j]=0\) for all \(j\)); this is enforced by the capacity constraint above.
    -   **Non-negativity:** \(x[i,j] \geq 0\) for all \(i, j\)
    -   **Binary:** \(y[i] \in \{0,1\}\) for all \(i\)
[Abstract Model Plan END]