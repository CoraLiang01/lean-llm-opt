#### Abstract Optimization Model

**Index Sets:**
- $\mathcal{C}$: Set of component types (from `unit_price.csv` and columns of `processing_time_unit.csv`)
- $\mathcal{W}$: Set of workshops (from rows of `processing_time_unit.csv` and `total_working_hours.csv`)

**Parameters:**
- $p_c$: Unit price of component $c \in \mathcal{C}$  
  (from `unit_price.csv`, column `unit_price`, indexed by `Unnamed: 0`)
- $t_{w,c}$: Unit processing time required for component $c$ in workshop $w$  
  (from `processing_time_unit.csv`, value at row `w`, column $c$)
- $H_w$: Total available working hours in workshop $w \in \mathcal{W}$  
  (from `total_working_hours.csv`, column `total_hours`, indexed by `workshop`)

**Decision Variables:**
- $x_c \geq 0$: Number of units to produce of component $c \in \mathcal{C}$ (continuous or integer, as appropriate)

**Objective:**
\[
\max \sum_{c \in \mathcal{C}} p_c \cdot x_c
\]

**Constraints:**
1. **Workshop Capacity Constraints:**  
   For each workshop $w \in \mathcal{W}$,
   \[
   \sum_{c \in \mathcal{C}} t_{w,c} \cdot x_c \leq H_w
   \]

2. **Non-negativity:**
   \[
   x_c \geq 0 \quad \forall c \in \mathcal{C}
   \]

---

#### Data Mapping

- $\mathcal{C}$: All columns except `Unnamed: 0` in `processing_time_unit.csv` (column names: `C1`, `C2`, ..., `C111`); all values of `Unnamed: 0` in `unit_price.csv`
- $\mathcal{W}$: All values of `Unnamed: 0` in `processing_time_unit.csv` (workshop names: `Casting`, `Milling`, `Finishing`, `Assembly`, `QA & Packaging`); all values of `workshop` in `total_working_hours.csv`
- $t_{w,c}$: Value at row with `Unnamed: 0` = $w$ and column $c$ in `processing_time_unit.csv`
- $p_c$: Value in `unit_price.csv`, column `unit_price`, where `Unnamed: 0` = $c$
- $H_w$: Value in `total_working_hours.csv`, column `total_hours`, where `workshop` = $w$

---

**Source Table IDs and Columns:**
- `processing_time_unit.csv` (`file_0_view_0`):  
  - Workshops: `Unnamed: 0`  
  - Components: columns `C1`–`C111`
- `unit_price.csv` (`file_1_view_0`):  
  - Components: `Unnamed: 0`  
  - Unit price: `unit_price`
- `total_working_hours.csv` (`file_2_view_0`):  
  - Workshops: `workshop`  
  - Total hours: `total_hours`