#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of service centres (from “Service Center” in service_centers_fixed_costs.csv; e.g., SC1–SC10)
- $C$: set of customers (from “Customer” in expanded_customer_service_costs.csv; e.g., C1–C15)

**Parameters:**
- $f_s$: fixed opening cost for centre $s \in S$ (from “Fixed Opening Cost” in service_centers_fixed_costs.csv)
- $c_{cs}$: cost to serve customer $c \in C$ from centre $s \in S$ (from column $s$ in expanded_customer_service_costs.csv, row $c$)

**Decision Variables:**
- $y_s \in \{0,1\}$: $1$ if centre $s$ is opened, $0$ otherwise
- $x_{cs} \in \{0,1\}$: $1$ if customer $c$ is assigned to centre $s$, $0$ otherwise

**Objective:**
\[
\min \quad \sum_{s \in S} f_s y_s + \sum_{c \in C} \sum_{s \in S} c_{cs} x_{cs}
\]

**Constraints:**
1. **Assignment:** Each customer is assigned to exactly one centre:
   \[
   \sum_{s \in S} x_{cs} = 1 \quad \forall c \in C
   \]
2. **Open centre for assignment:** Customers can only be assigned to opened centres:
   \[
   x_{cs} \leq y_s \quad \forall c \in C,\, s \in S
   \]
3. **Centre capacity:** Each centre serves at most 4 customers:
   \[
   \sum_{c \in C} x_{cs} \leq 4 \quad \forall s \in S
   \]
4. **Variable domains:**
   \[
   y_s \in \{0,1\} \quad \forall s \in S
   \]
   \[
   x_{cs} \in \{0,1\} \quad \forall c \in C,\, s \in S
   \]

---

**Data Mapping:**

- Table: service_centers_fixed_costs.csv
  - Index set $S$: column “Service Center”
  - Parameter $f_s$: column “Fixed Opening Cost”
- Table: expanded_customer_service_costs.csv
  - Index set $C$: column “Customer”
  - Parameter $c_{cs}$: columns “SC1”–“SC10” (one per $s \in S$), rows indexed by “Customer” (one per $c \in C$)