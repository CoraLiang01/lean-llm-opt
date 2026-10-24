{
  "query": "A small courier company operates a single delivery van that must depart from the depot, visit three customer locations — A, B, and C — exactly once each in any order, and then return to the depot on the same day. The pairwise road distances (in kilometres) between the depot and every location are provided in DistanceMatrix.csv. There are no service-time or time-window constraints. Formulate this problem and determine the sequence of visits that minimises the total travel distance.",
  "relationships": [
    {
      "column_axis": {
        "id_column": "Unnamed: 0",
        "table_id": "file_0_view_0"
      },
      "matrix_table_id": "file_0_view_0",
      "row_axis": {
        "id_column": "Unnamed: 0",
        "table_id": "file_0_view_0"
      },
      "row_id_column": "Unnamed: 0",
      "type": "matrix"
    }
  ],
  "tables": [
    {
      "columns": [
        "Unnamed: 0",
        "Depot",
        "A",
        "B",
        "C"
      ],
      "file_name": "DistanceMatrix.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "A": "28",
            "B": "41",
            "C": "63",
            "Depot": "0",
            "Unnamed: 0": "Depot"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "A": "0",
            "B": "27",
            "C": "87",
            "Depot": "28",
            "Unnamed: 0": "A"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "A": "27",
            "B": "0",
            "C": "81",
            "Depot": "41",
            "Unnamed: 0": "B"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "A": "87",
            "B": "81",
            "C": "0",
            "Depot": "63",
            "Unnamed: 0": "C"
          }
        }
      ],
      "table_id": "file_0_view_0"
    }
  ]
}
