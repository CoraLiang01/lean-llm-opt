Here are all rows and columns from parameters.csv, preserving truck identifiers and all parameter values as requested:

| truck_id | Q    | S   | C   | d1   | d2   | d3   | d4   |
|----------|------|-----|-----|------|------|------|------|
| 1        | 1000 | 500 | 2.0 | 1500 | 2000 | 1800 | 1000 |
| 2        | 800  | 300 | 3.0 | 1500 | 2000 | 1800 | 1000 |
| 3        | 1200 | 400 | 2.5 | 1500 | 2000 | 1800 | 1000 |
| 4        | 600  | 250 | 3.0 | 1500 | 2000 | 1800 | 1000 |
| 5        | 900  | 450 | 2.2 | 1500 | 2000 | 1800 | 1000 |
| 6        | 700  | 280 | 2.8 | 1500 | 2000 | 1800 | 1000 |
| 7        | 1100 | 420 | 2.4 | 1500 | 2000 | 1800 | 1000 |
| 8        | 500  | 200 | 3.2 | 1500 | 2000 | 1800 | 1000 |
| 9        | 1000 | 480 | 2.1 | 1500 | 2000 | 1800 | 1000 |
| 10       | 650  | 260 | 2.9 | 1500 | 2000 | 1800 | 1000 |

**Parameter meanings:**
- **truck_id**: Truck identifier (facility ID)
- **Q**: Maximum capacity (kg)
- **S**: Startup cost (fixed cost)
- **C**: Unit transportation cost (per kg)
- **d1, d2, d3, d4**: Customer demand in periods 1–4 (kg)

**Matrix orientation and shape:**
- Facilities (trucks): truck_id 1–10, each with Q, S, C
- Customers (periods): d1, d2, d3, d4 (periods 1–4)
- Cost-matrix axis: C (unit cost per truck per period)
- Demand axis: d1–d4 (customer demand per period)
- All data is preserved as in the source, with no transposition, truncation, or inference.