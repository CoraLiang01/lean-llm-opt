#### Symbolic Mathematical Model

Let $W$ be the set of all workers, as given by the columns (excluding "Owner") in file_0_view_0. Let $H$ be the set of all homeowners, as given by the "Owner" column in file_0_view_0. For this problem, $W = H$ (each worker is also a homeowner).

Let $d_{hw}$ be the number of days worker $w \in W$ worked on homeowner $h \in H$'s home, as given by file_0_view_0[Owner=h, w].

Let $r_w$ be the daily wage of worker $w \in W$ (decision variables).

Let $w_0$ be the first worker listed in the file (the first column after "Owner"), whose wage is fixed at 60.00.

Each worker contributes exactly 10 work days in total across all projects (including their own home).

The goal is to determine $r_w$ for all $w \in W$ so that, for every participant $h \in H$, their total income from working on others’ homes equals their total expenditure for work performed at their own home.

**Variables:**
- $r_w \in \mathbb{R}_{\geq 0}$, $\forall w \in W$ (daily wage of worker $w$)

**Parameters:**
- $d_{hw} \in \mathbb{Z}_{\geq 0}$: days worker $w$ worked on homeowner $h$'s home (from file_0_view_0)
- $w_0$: first worker column in file_0_view_0 (e.g., "Carpenter")
- $r_{w_0} = 60.00$

**Model:**

For all $h \in H$:
$$
\sum_{w \in W, w \neq h} d_{hw} \cdot r_w = \sum_{w \in W, w \neq h} d_{wh} \cdot r_h
$$

Or, equivalently, for all $h \in H$:
$$
\sum_{w \in W, w \neq h} d_{hw} \cdot r_w - \left(\sum_{w \in W, w \neq h} d_{wh}\right) \cdot r_h = 0
$$

With the normalization:
$$
r_{w_0} = 60.00
$$

**Domain:**
- $r_w \in \mathbb{R}_{\geq 0}$, $\forall w \in W$

#### Data Mapping

- Index set $W$ (workers): all columns except "Owner" in file_0_view_0
- Index set $H$ (homeowners): all values in "Owner" column in file_0_view_0
- Parameter $d_{hw}$: file_0_view_0[Owner=h, w]
- Variable $r_w$: daily wage of worker $w$
- Fixed wage: $r_{w_0} = 60.00$, where $w_0$ is the first worker column in file_0_view_0

**Note:** The model uses all rows and columns as required, and aligns all indices and parameters to the original file and column names.