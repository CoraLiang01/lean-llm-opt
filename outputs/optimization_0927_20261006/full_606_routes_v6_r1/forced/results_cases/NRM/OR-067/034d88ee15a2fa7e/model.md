#### Index Sets

- $M$: set of managers (from column "Unnamed: 0" in table_id file_0_view_0)
- $P$: set of projects (from columns "P1", "P2", "P3" in table_id file_0_view_0)

#### Parameters

- $c_{mp}$: cost for manager $m \in M$ to complete project $p \in P$ (from table_id file_0_view_0, columns "P1", "P2", "P3", rows indexed by "Unnamed: 0")

#### Decision Variables

- $x_{mp} \in \{0,1\}$: $1$ if manager $m$ is assigned to project $p$, $0$ otherwise

#### Objective

$$
\min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}
$$

#### Constraints

1. **Each manager is assigned to exactly one project:**
   $$
   \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
   $$

2. **Each project is assigned to exactly one manager:**
   $$
   \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
   $$

3. **Binary assignment:**
   $$
   x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
   $$

---

#### Data Mapping

- **Managers:** table_id file_0_view_0, column "Unnamed: 0"
- **Projects:** table_id file_0_view_0, columns "P1", "P2", "P3"
- **Costs $c_{mp}$:** table_id file_0_view_0, value at row manager $m$, column project $p$