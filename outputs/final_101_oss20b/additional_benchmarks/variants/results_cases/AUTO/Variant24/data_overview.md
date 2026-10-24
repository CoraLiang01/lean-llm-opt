{
  "query": "A community health program must choose facility sites and assign neighborhoods to opened sites. Neighborhood demand is listed in neighborhood_demand.csv, distances from candidate facilities to neighborhoods are listed in facility_distance.csv, and the number of facilities to open is listed in planning_parameters.csv. Each neighborhood must be assigned to exactly one opened facility.\n\nFormulate a minimum-demand-weighted-distance p-median model. For each candidate facility i, define y_i as a binary variable equal to 1 if facility i is opened. For each facility-neighborhood pair i-j, define x_ij as a binary variable equal to 1 if neighborhood j is assigned to facility i. The objective is to minimize total demand-weighted assignment distance. The model should include exactly-one assignment constraints for all neighborhoods, a constraint opening exactly the required number of facilities, assignment-to-open-facility linking constraints, and binary restrictions for all variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Neighborhood",
        "Demand"
      ],
      "file_name": "neighborhood_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Demand": "28",
            "Neighborhood": "N1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Demand": "32",
            "Neighborhood": "N2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Demand": "44",
            "Neighborhood": "N3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Demand": "36",
            "Neighborhood": "N4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Demand": "52",
            "Neighborhood": "N5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Demand": "41",
            "Neighborhood": "N6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Demand": "25",
            "Neighborhood": "N7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Demand": "48",
            "Neighborhood": "N8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Demand": "55",
            "Neighborhood": "N9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Demand": "30",
            "Neighborhood": "N10"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Facility",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "N7",
        "N8",
        "N9",
        "N10"
      ],
      "file_name": "facility_distance.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Facility": "F1",
            "N1": "2",
            "N10": "12",
            "N2": "3",
            "N3": "5",
            "N4": "9",
            "N5": "10",
            "N6": "11",
            "N7": "13",
            "N8": "14",
            "N9": "15"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Facility": "F2",
            "N1": "4",
            "N10": "11",
            "N2": "2",
            "N3": "3",
            "N4": "8",
            "N5": "9",
            "N6": "10",
            "N7": "12",
            "N8": "13",
            "N9": "14"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Facility": "F3",
            "N1": "9",
            "N10": "7",
            "N2": "8",
            "N3": "4",
            "N4": "2",
            "N5": "3",
            "N6": "5",
            "N7": "9",
            "N8": "10",
            "N9": "11"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Facility": "F4",
            "N1": "10",
            "N10": "6",
            "N2": "9",
            "N3": "6",
            "N4": "3",
            "N5": "2",
            "N6": "3",
            "N7": "8",
            "N8": "9",
            "N9": "10"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Facility": "F5",
            "N1": "13",
            "N10": "5",
            "N2": "12",
            "N3": "10",
            "N4": "8",
            "N5": "7",
            "N6": "6",
            "N7": "2",
            "N8": "3",
            "N9": "4"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Facility": "F6",
            "N1": "12",
            "N10": "2",
            "N2": "11",
            "N3": "8",
            "N4": "7",
            "N5": "6",
            "N6": "5",
            "N7": "5",
            "N8": "4",
            "N9": "3"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "Parameter",
        "Value"
      ],
      "file_name": "planning_parameters.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Parameter": "NumberOfFacilitiesToOpen",
            "Value": "2"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
