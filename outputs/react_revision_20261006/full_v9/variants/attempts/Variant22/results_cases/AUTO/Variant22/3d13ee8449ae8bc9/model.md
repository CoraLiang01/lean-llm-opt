## Symbolic Mathematical Model

**Sets**
- $C$: set of component families (from option_catalog.csv, column "Family")
- $O_c$: set of options for family $c \in C$ (from option_catalog.csv, column "Option" for each $c$)
- $R$: set of resources (from resource_limits.csv, column "Resource")

**Parameters**
- $v_{co}$: value of option $o \in O_c$ in family $c$ (option_catalog.csv, "Value")
- $w_{co}$: weight of option $o \in O_c$ in family $c$ (option_catalog.csv, "Weight")
- $l_{co}$: labor-hours of option $o \in O_c$ in family $c$ (option_catalog.csv, "LaborHours")
- $W^{\max}$: total weight limit (resource_limits.csv, row with Resource="Weight", column "Limit")
- $L^{\max}$: total labor-hour limit (resource_limits.csv, row with Resource="LaborHours", column "Limit")

**Decision Variables**
- $x_{co} \in \{0,1\}$: 1 if option $o$ is selected from family $c$, 0 otherwise

**Objective**
\[
\max \sum_{c \in C} \sum_{o \in O_c} v_{co} x_{co}
\]

**Constraints**
1. **Exactly one option per family:**
   \[
   \sum_{o \in O_c} x_{co} = 1 \quad \forall c \in C
   \]
2. **Total weight limit:**
   \[
   \sum_{c \in C} \sum_{o \in O_c} w_{co} x_{co} \leq W^{\max}
   \]
3. **Total labor-hour limit:**
   \[
   \sum_{c \in C} \sum_{o \in O_c} l_{co} x_{co} \leq L^{\max}
   \]
4. **Binary variables:**
   \[
   x_{co} \in \{0,1\} \quad \forall c \in C,\, o \in O_c
   \]

---

## Data Mapping

- $C$ and $O_c$ are defined by all unique "Family" and "Option" pairs in option_catalog.csv (table_id: file_0_view_0, columns: "Family", "Option").
- $v_{co}$, $w_{co}$, $l_{co}$ are from option_catalog.csv (table_id: file_0_view_0, columns: "Value", "Weight", "LaborHours").
- $W^{\max}$ and $L^{\max}$ are from resource_limits.csv (table_id: file_1_view_0, rows with Resource="Weight" and "LaborHours", column "Limit").
- $x_{co}$ is defined for every $(c,o)$ pair in option_catalog.csv.