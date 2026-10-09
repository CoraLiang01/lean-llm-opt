## Mathematical Model

**Sets**
- $O$: set of all generation options (option), from table_id: file_0_view_0

**Parameters (from file_0_view_0)**
- $c_o$: cost per lot for option $o$ (cost_per_lot)
- $g_o$: generation per lot for option $o$ (gen_per_lot)
- $D$: total demand $= 200$

**Decision Variables**
- $x_o \in \mathbb{Z}_+$: number of lots to purchase for option $o \in O$

**Objective**
$$
\min \sum_{o \in O} c_o\, x_o
$$

**Constraint**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$

$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

**Data Mapping**

- $O$: All rows in file_0_view_0, column "option"
- $c_o$: file_0_view_0, column "cost_per_lot", for each $o$
- $g_o$: file_0_view_0, column "gen_per_lot", for each $o$
- $D$: 200 (from user description)
- $x_o$: integer variable for each $o \in O$