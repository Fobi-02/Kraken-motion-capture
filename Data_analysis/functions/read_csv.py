import csv
import pandas as pd


def read_csv(file_path):
    """
    reads a CSV file of the mocap data and returns a dataframe
    """

    # reading csv
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # ROWS FORMAT:
    # Metadata
    # Type
    # Name
    # ID
    # Position
    # Frame, Time (Seconds), X, Y, Z, ...

    # searching the line that starts with Frame
    header_index = None

    for i, row in enumerate(rows):
        if len(row) > 0 and row[0].strip() == "Frame":
            header_index = i
            break

    if header_index is None:
        raise ValueError("No rows found matching the motion capture CSV format.")

    # saving the type and the name of the columns
    row_name = rows[header_index - 3]
    row_data_type = rows[header_index - 1]

    # deciding the names of the columns
    columns = []
    columns.append("Frame")
    columns.append("Time")

    index = 2

    while index < len(row_name):
        marker_name = row_name[index].strip()
        if not marker_name:
            break
        # names as p_1501_X
        if marker_name.startswith("Unlabeled "):
            marker_number = marker_name.split(" ")[1]
            marker_name = f"p_{marker_number}"
        columns.append(f"{marker_name}_X")
        columns.append(f"{marker_name}_Y")
        columns.append(f"{marker_name}_Z")
        index += 3

    # Reading the data in the csv
    data = []
    for row in rows[header_index + 1:]:
        # checking if the format is correct
        if not row:
            continue
        if not row[0].strip():
            continue
        data.append(row)

    # creating the data frame
    df = pd.DataFrame(data,columns=columns)

    # converting numerical values
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df