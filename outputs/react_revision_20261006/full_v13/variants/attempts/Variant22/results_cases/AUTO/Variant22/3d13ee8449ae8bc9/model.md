#### Mathematical Model

**Index Sets:**
- $\mathcal{C}$: set of component families (Family from file_0_view_0)
- $\mathcal{O}_c$: set of options available for family $c \in \mathcal{C}$ (Option from file_0_view_0 for each Family)

**Parameters:**
- $v_{co}$: value of option $o$ in family $c$ (Value from file_0_view_0)
- $w_{co}$: weight of option $o$ in family $c$ (Weight from file_0_view_0)
- $l_{co}$: labor-hours of option $o$ in family $c$ (LaborHours from file_0_view_0)
- $W^{\max}$: total weight limit (Limit where Resource = "Weight" from file_1_view_0)
- $L^{\max}$: total labor-hour limit (Limit where Resource = "LaborHours" from file_1_view_0)

**Decision Variables:**
- $x_{co} \in \{0,1\}$: $=1$ if option $o$ is selected from family $c$, $0$ otherwise

**Objective:**
\[
\max \sum_{c \in \mathcal{C}} \sum_{o \in \mathcal{O}_c} v_{co} x_{co}
\]

**Constraints:**
1. **Exactly-one-option per family:**
   \[
   \sum_{o \in \mathcal{O}_c} x_{co} = 1 \qquad \forall c \in \mathcal{C}
   \]
2. **Total weight limit:**
   \[
   \sum_{c \in \mathcal{C}} \sum_{o \in \mathcal{O}_c} w_{co} x_{co} \leq W^{\max}
   \]
3. **Total labor-hour limit:**
   \[
   \sum_{c \in \mathcal{C}} \sum_{o \in \mathcal{O}_c} l_{co} x_{co} \leq L^{\max}
   \]
4. **Binary restrictions:**
   \[
   x_{co} \in \{0,1\} \qquad \forall c \in \mathcal{C},\ o \in \mathcal{O}_c
   \]

---

#### Data Mapping

- $\mathcal{C}$: All unique values in `Family` column of `file_0_view_0` (option_catalog.csv)
- $\mathcal{O}_c$: All `Option` values in `file_0_view_0` where `Family` = $c$
- $v_{co}$: `Value` column in `file_0_view_0` for each (Family, Option) pair
- $w_{co}$: `Weight` column in `file_0_view_0` for each (Family, Option) pair
- $l_{co}$: `LaborHours` column in `file_0_view_0` for each (Family, Option) pair
- $W^{\max}$: `Limit` in `file_1_view_0` where `Resource` = "Weight" (resource_limits.csv)
- $L^{\max}$: `Limit` in `file_1_view_0` where `Resource` = "LaborHours" (resource_limits.csv)