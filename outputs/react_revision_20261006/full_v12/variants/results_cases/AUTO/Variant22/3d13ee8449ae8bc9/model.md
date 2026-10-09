## Symbolic Mathematical Model

**Sets**  
$C$ = set of component families (from option_catalog.csv, column Family)  
$O_c$ = set of options for family $c \in C$ (from option_catalog.csv, column Option for each Family)

**Parameters**  
$v_{c,o}$ = value of option $o$ in family $c$ (option_catalog.csv, Value)  
$w_{c,o}$ = weight of option $o$ in family $c$ (option_catalog.csv, Weight)  
$l_{c,o}$ = labor-hours of option $o$ in family $c$ (option_catalog.csv, LaborHours)  
$W^{\max}$ = total weight limit (resource_limits.csv, Resource = Weight, Limit)  
$L^{\max}$ = total labor-hour limit (resource_limits.csv, Resource = LaborHours, Limit)

**Decision Variables**  
$x_{c,o} \in \{0,1\}$, for all $c \in C$, $o \in O_c$  
($x_{c,o} = 1$ if option $o$ is selected from family $c$, 0 otherwise)

**Objective**  
$\max \sum_{c \in C} \sum_{o \in O_c} v_{c,o} x_{c,o}$

**Constraints**  
1. **Exactly one option per family:**  
$\sum_{o \in O_c} x_{c,o} = 1 \qquad \forall c \in C$

2. **Total weight limit:**  
$\sum_{c \in C} \sum_{o \in O_c} w_{c,o} x_{c,o} \leq W^{\max}$

3. **Total labor-hour limit:**  
$\sum_{c \in C} \sum_{o \in O_c} l_{c,o} x_{c,o} \leq L^{\max}$

4. **Binary variables:**  
$x_{c,o} \in \{0,1\} \qquad \forall c \in C,\, o \in O_c$

---

### Data Mapping

- $C$ and $O_c$ from option_catalog.csv:  
  - table_id: file_0_view_0, columns: Family, Option
- $v_{c,o}$ from option_catalog.csv:  
  - table_id: file_0_view_0, column: Value
- $w_{c,o}$ from option_catalog.csv:  
  - table_id: file_0_view_0, column: Weight
- $l_{c,o}$ from option_catalog.csv:  
  - table_id: file_0_view_0, column: LaborHours
- $W^{\max}$ from resource_limits.csv:  
  - table_id: file_1_view_0, Resource = Weight, column: Limit
- $L^{\max}$ from resource_limits.csv:  
  - table_id: file_1_view_0, Resource = LaborHours, column: Limit