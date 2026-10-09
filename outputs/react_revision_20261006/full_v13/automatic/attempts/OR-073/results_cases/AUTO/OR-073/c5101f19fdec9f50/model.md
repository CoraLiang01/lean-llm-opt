## Symbolic Mathematical Model

### Sets
- $P = \{\text{I}, \text{II}, \text{III}\}$: Products
- $A = \{\text{A1}, \text{A2}\}$: Equipment for procedure A
- $B = \{\text{B1}, \text{B2}, \text{B3}\}$: Equipment for procedure B

### Parameters (from Data Mapping)
Let $t_{e,p}$ = processing time per unit of product $p$ on equipment $e$ (minutes/unit)  
Let $T_e$ = available operating time for equipment $e$ (minutes)  
Let $C_e$ = equipment cost at full load for equipment $e$ (yuan)  
Let $c_p$ = raw material cost per unit of product $p$ (yuan/unit)  
Let $s_p$ = selling price per unit of product $p$ (yuan/unit)

#### Data Mapping (table_id: file_0_view_0)
- $t_{e,p}$: "Product I", "Product II", "Product III" columns, for each equipment row
- $T_e$: "Available Equipment Operating Time" column
- $C_e$: "Equipment Cost at Full Load (yuan)" column
- $c_p$: "Raw Material Cost (yuan/unit)" row
- $s_p$: "Unit Price (yuan/unit)" row

### Decision Variables
- $x_{A,e,p} \geq 0$: quantity of product $p$ processed on equipment $e$ in procedure A
- $x_{B,e,p} \geq 0$: quantity of product $p$ processed on equipment $e$ in procedure B

### Objective
Maximize total profit:
\[
\max \left\{
\sum_{p \in P} s_p \cdot y_p
- \sum_{p \in P} c_p \cdot y_p
- \sum_{e \in A \cup B} \frac{C_e}{T_e} \cdot \sum_{p \in P} t_{e,p} \cdot z_{e,p}
\right\}
\]
where:
- $y_p$ = total quantity of product $p$ produced and sold
- $z_{e,p}$ = total units of product $p$ processed on equipment $e$ (in its respective procedure)
- For each $p$, $y_p$ is the total flow through both procedures (see below).

### Constraints

#### 1. Equipment-Product Assignment (from process and equipment compatibility)
- For procedure A:
    - Product I: $x_{A,\text{A1},\text{I}},\ x_{A,\text{A2},\text{I}}$ allowed
    - Product II: $x_{A,\text{A1},\text{II}},\ x_{A,\text{A2},\text{II}}$ allowed
    - Product III: $x_{A,\text{A2},\text{III}}$ only
- For procedure B:
    - Product I: $x_{B,\text{B1},\text{I}},\ x_{B,\text{B2},\text{I}},\ x_{B,\text{B3},\text{I}}$ allowed
    - Product II: $x_{B,\text{B1},\text{II}}$ only
    - Product III: $x_{B,\text{B2},\text{III}}$ only

#### 2. Flow Conservation (each product must be processed by both procedures, so output of A = input to B for each product)
\[
\sum_{e \in A_p} x_{A,e,p} = \sum_{e \in B_p} x_{B,e,p} = y_p \qquad \forall p \in P
\]
where:
- $A_{\text{I}} = \{\text{A1}, \text{A2}\}$, $A_{\text{II}} = \{\text{A1}, \text{A2}\}$, $A_{\text{III}} = \{\text{A2}\}$
- $B_{\text{I}} = \{\text{B1}, \text{B2}, \text{B3}\}$, $B_{\text{II}} = \{\text{B1}\}$, $B_{\text{III}} = \{\text{B2}\}$

#### 3. Equipment Time Capacity
For all $e \in A \cup B$:
\[
\sum_{p \in P_e} t_{e,p} \cdot z_{e,p} \leq T_e
\]
where $P_e$ is the set of products that can be processed on equipment $e$ in its respective procedure, and $z_{e,p}$ is $x_{A,e,p}$ if $e \in A$, $x_{B,e,p}$ if $e \in B$.

#### 4. Non-negativity
\[
x_{A,e,p} \geq 0,\quad x_{B,e,p} \geq 0 \qquad \forall\ e,p\ \text{allowed}
\]

### Data Mapping

- $t_{e,p}$: file_0_view_0, columns "Product I", "Product II", "Product III", for each equipment row (A1, A2, B1, B2, B3)
- $T_e$: file_0_view_0, column "Available Equipment Operating Time", for each equipment row
- $C_e$: file_0_view_0, column "Equipment Cost at Full Load (yuan)", for each equipment row
- $c_p$: file_0_view_0, row "Raw Material Cost (yuan/unit)", columns "Product I", "Product II", "Product III"
- $s_p$: file_0_view_0, row "Unit Price (yuan/unit)", columns "Product I", "Product II", "Product III"

### Explicit Variable and Constraint List

#### Variables
- $x_{A,\text{A1},\text{I}} \geq 0$
- $x_{A,\text{A2},\text{I}} \geq 0$
- $x_{A,\text{A1},\text{II}} \geq 0$
- $x_{A,\text{A2},\text{II}} \geq 0$
- $x_{A,\text{A2},\text{III}} \geq 0$
- $x_{B,\text{B1},\text{I}} \geq 0$
- $x_{B,\text{B2},\text{I}} \geq 0$
- $x_{B,\text{B3},\text{I}} \geq 0$
- $x_{B,\text{B1},\text{II}} \geq 0$
- $x_{B,\text{B2},\text{III}} \geq 0$

#### Constraints
- $x_{A,\text{A1},\text{I}} + x_{A,\text{A2},\text{I}} = x_{B,\text{B1},\text{I}} + x_{B,\text{B2},\text{I}} + x_{B,\text{B3},\text{I}} = y_{\text{I}}$
- $x_{A,\text{A1},\text{II}} + x_{A,\text{A2},\text{II}} = x_{B,\text{B1},\text{II}} = y_{\text{II}}$
- $x_{A,\text{A2},\text{III}} = x_{B,\text{B2},\text{III}} = y_{\text{III}}$
- $\sum_{p} t_{\text{A1},p} \cdot x_{A,\text{A1},p} \leq T_{\text{A1}}$ (only for allowed $p$)
- $\sum_{p} t_{\text{A2},p} \cdot x_{A,\text{A2},p} \leq T_{\text{A2}}$ (only for allowed $p$)
- $\sum_{p} t_{\text{B1},p} \cdot x_{B,\text{B1},p} \leq T_{\text{B1}}$ (only for allowed $p$)
- $\sum_{p} t_{\text{B2},p} \cdot x_{B,\text{B2},p} \leq T_{\text{B2}}$ (only for allowed $p$)
- $\sum_{p} t_{\text{B3},p} \cdot x_{B,\text{B3},p} \leq T_{\text{B3}}$ (only for allowed $p$)

#### Objective (expanded)
\[
\max \left\{
\sum_{p \in P} s_p y_p
- \sum_{p \in P} c_p y_p
- \sum_{e \in A} \frac{C_e}{T_e} \sum_{p \in P_e} t_{e,p} x_{A,e,p}
- \sum_{e \in B} \frac{C_e}{T_e} \sum_{p \in P_e} t_{e,p} x_{B,e,p}
\right\}
\]

where all parameters and variable indices are mapped as above.

---

**Data Mapping:**  
All parameters are mapped to file_0_view_0 (43.csv) as described above.  
All variable and constraint indices are determined by the product-equipment compatibility in the user description and the data.  
No additional constraints or variables are introduced beyond those required by the question and data.