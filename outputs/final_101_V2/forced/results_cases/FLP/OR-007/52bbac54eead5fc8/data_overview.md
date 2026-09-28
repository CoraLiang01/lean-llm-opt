**customer_demand.csv**

| customer | demand |
|----------|--------|
| D1       | 428    |
| D2       | 217    |
| D3       | 214    |
| D4       | 380    |
| D5       | 254    |

---

**supply_capacity.csv**

| region | supply_capacity |
|--------|----------------|
| S1     | 428            |
| S2     | 217            |
| S3     | 214            |
| S4     | 380            |
| S5     | 254            |

---

**transportation_costs.csv**

|        | D1                  | D2                  | D3                  | D4                  | D5                  |
|--------|---------------------|---------------------|---------------------|---------------------|---------------------|
| S1     | 269.3910588020795   | 1.4537335390933939  | 99.60345345756605   | 26.64078166309837   | 9.537688956880922   |
| S2     | 9.291846876785183   | 10.874778437070223  | 144.52609291614627  | 11.420133077898234  | 153.1756819927813   |
| S3     | 9.674584301671008   | 2.6191650959687944  | 100.8242249168735   | 3.2121910887916876  | 133.8493396124168   |
| S4     | 270.57498480010247  | 32.50253586         | 4.6842098096469815  | 1.5682269686546804  | 9.58927599          |
| S5     | 226.0331910675782   | 8.669161980826471   | 65.47681316968448   | 9.068765258459958   | 202.65015316425533  |

---

**Preserved Identifiers and Source Orientation:**

- Facility IDs (warehouses): S1, S2, S3, S4, S5
- Customer IDs (stores): D1, D2, D3, D4, D5
- Each supply_capacity is associated with its facility ID.
- Each demand is associated with its customer ID.
- The transportation_costs matrix is indexed by facility (row: S1–S5) and customer (column: D1–D5), preserving original orientation and shape.