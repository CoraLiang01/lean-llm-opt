#### Index Sets

- $\mathcal{S}$: Set of service centres (from Service Center in service_centers_fixed_costs.csv)
- $\mathcal{C}$: Set of customers (from Customer in expanded_customer_service_costs.csv)

#### Parameters

- $f_s$: Fixed opening cost for centre $s \in \mathcal{S}$ (from Fixed Opening Cost in service_centers_fixed_costs.csv)
- $c_{cs}$: Service cost to assign customer $c \in \mathcal{C}$ to centre $s \in \mathcal{S}$ (from SC columns in expanded_customer_service_costs.csv)

#### Decision Variables

- $y_s \in \{0,1\}$: 1 if centre $s$ is opened, 0 otherwise
- $x_{cs} \in \{0,1\}$: 1 if customer $c$ is assigned to centre $s$, 0 otherwise

#### Objective

$$
\min \quad \sum_{s \in \mathcal{S}} f_s y_s + \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}} c_{cs} x_{cs}
$$

#### Constraints

1. **Each customer assigned to exactly one centre:**
   $$
   \sum_{s \in \mathcal{S}} x_{cs} = 1 \quad \forall c \in \mathcal{C}
   $$

2. **Assignment only to open centres:**
   $$
   x_{cs} \leq y_s \quad \forall c \in \mathcal{C},\ s \in \mathcal{S}
   $$

3. **Each centre serves at most 4 customers:**
   $$
   \sum_{c \in \mathcal{C}} x_{cs} \leq 4 \quad \forall s \in \mathcal{S}
   $$

4. **Variable domains:**
   $$
   y_s \in \{0,1\} \quad \forall s \in \mathcal{S}
   $$
   $$
   x_{cs} \in \{0,1\} \quad \forall c \in \mathcal{C},\ s \in \mathcal{S}
   $$

---

#### Data Mapping

- $\mathcal{S}$: All values in column Service Center of table_id file_1_view_0 (service_centers_fixed_costs.csv)
- $\mathcal{C}$: All values in column Customer of table_id file_0_view_0 (expanded_customer_service_costs.csv)
- $f_s$: Fixed Opening Cost column in table_id file_1_view_0, indexed by Service Center
- $c_{cs}$: SC1–SC10 columns in table_id file_0_view_0, indexed by Customer (rows) and Service Center (columns)