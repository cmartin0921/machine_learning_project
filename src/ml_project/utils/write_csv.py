import csv
from typing import Union, List, Dict, TextIO
from pathlib import Path

def write_csv(
    file_handle: TextIO,
    file_location: Path,
    data: Union[Dict, List[Dict], List[List], List],
    write_header: bool,
    file_header: List[str] = None
) -> str:

    if data is None or len(data) == 0:
        return f"No data passed to write in {file_location}"

    try:
        if isinstance(data, dict):
            writer = csv.DictWriter(file_handle, fieldnames=list(data.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(data)
        # Handling instances where there is more than a single row of data
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                writer = csv.DictWriter(file_handle, fieldnames=list(data[0].keys()))
                if write_header:
                    writer.writeheader()
                writer.writerows(data)
            elif isinstance(data[0], list):
                writer = csv.writer(file_handle)
                if write_header and file_header is not None:
                    writer.writerow(file_header)
                elif write_header and file_header is None:
                    raise ValueError("Header for lists in a newly created file is required.")
                writer.writerows(data)
        elif isinstance(data, list):
            writer = csv.writer(file_handle)
            if write_header and file_header is not None:
                writer.writerow(file_header)
            elif write_header and file_header is None:
                raise ValueError("Header for lists in a newly created file is required.")
            writer.writerow(data)
        else:
            return f"Unsupported data type: {type(data)}"
    except Exception as e:
        raise e
    
    return f"Successfully wrote data to {file_location}"



def write_csv_v2(
    file_handle: TextIO,
    data: Union[Dict, List[Dict], List[List]],
    file_header: List[str] = None
) -> bool:

    if not data:
        return False

    rows = [data] if isinstance(data, dict) else data
    first_row = rows[0]
    
    is_empty_file = file_handle.tell() == 0

    try:
        if isinstance(first_row, dict):
            writer = csv.DictWriter(file_handle, fieldnames=list(first_row.keys()))
            if is_empty_file:
                writer.writeheader()
            writer.writerows(rows)
            
        else:
            writer = csv.writer(file_handle)
            if is_empty_file:
                if file_header:
                    writer.writerow(file_header)
                else:
                    # Only raise if it's a list-of-lists and we have no keys to infer from
                    raise ValueError("File is empty and no file_header provided for list data.")
            writer.writerows(rows)
            
        return True
    except Exception as e:
        # Log the error here or re-raise
        raise e