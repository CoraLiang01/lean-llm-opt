## Mathematical Model

**Sets**
- $O$: Set of all generation options (from energy.csv, column "option")
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ (coal, gas, renewables)
    - $g_o$: Generation per lot for option $o$ (from "gen_per_lot")
    - $c_o$: Cost per lot for option $o$ (from "cost_per_lot")

**Parameters**
- $D = 200$: Total demand to be met

**Decision Variables**
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o \in O$

**Objective**
$$
\min \sum_{o \in O} c_o\, x_o
$$

**Constraints**
1. **Demand Satisfaction**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$

2. **Integrality**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

## Data Mapping

- $O$: All rows in energy.csv, column "option" (table_id: file_0_view_0, column: option)
- $tech_o$: energy.csv, column "tech" (table_id: file_0_view_0, column: tech)
- $g_o$: energy.csv, column "gen_per_lot" (table_id: file_0_view_0, column: gen_per_lot)
- $c_o$: energy.csv, column "cost_per_lot" (table_id: file_0_view_0, column: cost_per_lot)
- $D$: Demand = 200 (from user description)
- $x_o$: Integer variable, number of lots of option $o$ to purchase

All generation options in the file are available for selection. Each $x_o$ must be a nonnegative integer. The objective is to minimize total cost while meeting or exceeding the total demand.