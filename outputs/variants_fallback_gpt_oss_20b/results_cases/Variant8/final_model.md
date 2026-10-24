- Binary variables: \(y_i = 1\) if clinic \(i\) is opened, 0 otherwise.
- Binary variables: \(x_{ij} = 1\) if neighborhood \(j\) is assigned to clinic \(i\), 0 otherwise.
- Objective: minimize \(\sum_{i}\sum_{j} \text{demand}_j \times \text{distance}_{ij} \times x_{ij}\).
- Assignment: each neighborhood is assigned to exactly one clinic: \(\sum_i x_{ij} = 1\) for all \(j\).
- Linking: an assignment can only occur if the clinic is opened: \(x_{ij} \le y_i\) for all \(i,j\).
- Open count: exactly the required number of clinics are opened: \(\sum_i y_i = \text{NumberOfClinicsToOpen}\).

CODE: