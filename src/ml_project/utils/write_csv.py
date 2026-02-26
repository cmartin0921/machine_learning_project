import csv
from typing import Union, List, Dict, TextIO

def write_csv(
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
                    raise ValueError("File is empty and no file_header provided for list data.")
            writer.writerows(rows)

        return True
    except Exception as e:
        raise e
