**Mathematical Model**

Let  

* \(W=\{S1,S2,S3,S4,S5\}\) – set of warehouses (rows of *supply_capacity.csv*).  
* \(C=\{D1,D2,D3,D4,D5\}\) – set of stores (rows of *customer_demand.csv*).  

Parameters (all obtained directly from the supplied tables)

| Symbol | Definition | Source |
|--------|------------|--------|
| \(d_c\) | Daily demand of store \(c\) | *customer_demand.csv*, column **demand** |
| \(s_w\) | Daily supply capacity of warehouse \(w\) | *supply_capacity.csv*, column **supply_capacity** |
| \(t_{w,c}\) | Transportation cost per unit from warehouse \(w\) to store \(c\) | *transportation_costs.csv*, matrix entry \((w,c)\) |

Decision variables  

\[
x_{w,c}\ge 0 \quad \forall w\in W,\;c\in C
\]

*\(x_{w,c}\)* – quantity shipped from warehouse \(w\) to store \(c\).

**Objective**

\[
\min \; Z = \sum_{w\in W}\sum_{c\in C} t_{w,c}\,x_{w,c}
\]

**Constraints**

1. **Demand satisfaction** – each store’s demand must be met exactly  

\[
\sum_{w\in W} x_{w,c} = d_c \quad \forall c\in C
\]

2. **Supply capacity** – a warehouse cannot ship more than its capacity  

\[
\sum_{c\in C} x_{w,c} \le s_w \quad \forall w\in W
\]

3. **Non‑negativity**  

\[
x_{w,c}\ge 0 \quad \forall w\in W,\;c\in C
\]

---

### Data Mapping Rules

| Parameter | Table ID | Column(s) | Mapping |
|-----------|----------|-----------|---------|
| \(d_c\) | `file_0_view_0` | `customer`, `demand` | For each record, map `customer` → \(c\), `demand` → \(d_c\). |
| \(s_w\) | `file_1_view_0` | `region`, `supply_capacity` | For each record, map `region` → \(w\), `supply_capacity` → \(s_w\). |
| \(t_{w,c}\) | `file_2_view_0` | matrix with rows `Unnamed: 0` (warehouse) and columns `D1`–`D5` (stores) | For each row \(w\) and column \(c\), map the cell value to \(t_{w,c}\). |

All index sets \(W\) and \(C\) are taken from the full set of rows/columns in the respective tables; no additional filtering is applied. The model uses continuous shipment quantities, as specified.