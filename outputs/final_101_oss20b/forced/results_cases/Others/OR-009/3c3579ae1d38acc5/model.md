[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities from each production plant to each retail outlet so that all outlet demands are satisfied, no plant’s production capacity is exceeded, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a classic **Transportation Problem**, a linear programming (LP) model.

3.  **Define Index Sets:**  
   - **Plants** \(i \in \mathcal{S} = \{S1, S2, S3, S4\}\) (extracted from the “Unnamed: 0” column of *supply_capacity.csv*).  
   - **Customers** \(j \in \mathcal{C} = \{C1, C2, C3, C4\}\) (extracted from the “customer” column of *customer_demand.csv*).

4.  **Define Decision Variables:**  
   - `x[i,j]` = quantity of beverage shipped from plant \(i\) to customer \(j\).  
     Type: **GRB.CONTINUOUS** (non‑negative).  
   - No binary or integer variables are required.

5.  **Identify Parameters (from Schema):**  
   - **Demand**: `demand[j]` from *customer_demand.csv* (column “demand”).  
   - **Supply Capacity**: `supply_capacity[i]` from *supply_capacity.csv* (column “supply_capacity”).  
   - **Transportation Cost**: `cost[i,j]` from *transportation_costs.csv* (row “Unnamed: 0” gives plant \(i\); columns “C1”–“C4” give customer \(j\)).

6.  **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{i \in \mathcal{S}}\sum_{j \in \mathcal{C}} \text{cost}[i,j] \times x[i,j].
   \]

7.  **Formulate Constraints:**  
   - **Demand Satisfaction (for each customer \(j\))**  
     \[
     \sum_{i \in \mathcal{S}} x[i,j] = \text{demand}[j].
     \]
   - **Supply Capacity (for each plant \(i\))**  
     \[
     \sum_{j \in \mathcal{C}} x[i,j] \le \text{supply_capacity}[i].
     \]
   - **Non‑negativity**  
     \[
     x[i,j] \ge 0 \quad \forall i \in \mathcal{S},\, j \in \mathcal{C}.
     \]
[Abstract Model Plan END]