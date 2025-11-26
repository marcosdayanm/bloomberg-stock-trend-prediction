import pandas as pd
from pathlib import Path

from src.constants import CRUDE_DATASETS_DIR


def _excel_serial_to_date(series):
    """Turns from Excel serial date to pd.Timestamp."""
    return pd.to_datetime(series, unit="D", origin="1899-12-30")


def _is_excel_date_column(col, min_non_na=5):
    """Heuristic to detect Excel serial date columns."""
    col = pd.to_numeric(col, errors="coerce")
    non_na = col.notna().sum()
    if non_na < min_non_na:
        return False
    s = col.dropna()
    return ((s > 30000) & (s < 60000)).mean() > 0.8


def _is_numeric_column(col, min_non_na=5):
    """Numeric column detection."""
    col = pd.to_numeric(col, errors="coerce")
    return col.notna().sum() >= min_non_na


def _parse_indicator_block(df: pd.DataFrame, header_row: int, search_start_col=0):
    """
    This function parses a DataFrame to extract time series data
    organized in pairs of columns: [date, value], starting from a specified
    header row and column.
    """
    n_cols = df.shape[1]
    data_start = header_row + 1
    all_series = []

    c = search_start_col
    while c < n_cols - 1:
        field_code = df.iat[header_row, c]

        if pd.isna(field_code):
            c += 1
            continue

        date_col = df.iloc[data_start:, c]
        val_col = df.iloc[data_start:, c + 1]

        # pattern
        if not (_is_excel_date_column(date_col) and _is_numeric_column(val_col)):
            c += 1
            continue

        mask = date_col.notna() & val_col.notna()
        if not mask.any():
            c += 2
            continue

        dates = _excel_serial_to_date(date_col[mask].astype(float))
        values = pd.to_numeric(val_col[mask], errors="coerce")

        s = pd.Series(values.to_list(), index=dates, name=str(field_code))
        all_series.append(s)

        c += 2 

    if not all_series:
        return pd.DataFrame()

    out = pd.concat(all_series, axis=1)
    out = out.sort_index().reset_index().rename(columns={"index": "date"})
    return out


def process_workbook(excel_path: Path, header_row: int, out_dir: Path, search_start_col=0, ignore_sheets=set()):
    """
    This function processes an Excel workbook, extracting tables from each sheet
    that match a specific pattern and saving them as CSV files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(excel_path)

    for sheet in xls.sheet_names:
        if sheet.split("_")[0] in ignore_sheets or sheet.split("_")[1] == "BLOOMBERG":  # type: ignore
            print(f"[INFO] Ignoring sheet: {sheet}")
            continue
        df = xls.parse(sheet, header=None)
        table = _parse_indicator_block(
            df,
            header_row=header_row,
            search_start_col=search_start_col
        )

        if table.empty:
            print(f"[WARNING] Sheet '{sheet}' does not match the pattern.")
            continue

        base = sheet.replace(" ", "_")  # type: ignore
        csv_path = out_dir / f"{base}.csv"
        table.to_csv(csv_path, index=False)
        print(f"Saving: {csv_path}")


def get_latest_crude_dataset_xslx() -> Path:
    """Gets the latest Excel dataset from the crude datasets directory."""
    all_files = list(CRUDE_DATASETS_DIR.glob("*.xlsx"))

    if not all_files:
        raise FileNotFoundError(f"Unable to find files in {CRUDE_DATASETS_DIR} that match *.xlsx")

    all_files.sort(reverse=True)
    return all_files[0]


def create_crude():
    """Creates a 'CRUDE' csv file from all csv files in the crude datasets directory.
    Each column will have as a prefix the original file name.
    """
    all_csv_files = list(CRUDE_DATASETS_DIR.glob("*.csv"))

    if not all_csv_files:
        raise FileNotFoundError(f"Unable to find files in {CRUDE_DATASETS_DIR} that match *.csv")

    all_dfs = []
    for csv_path in all_csv_files:
        if csv_path.stem == "CRUDE":
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"])
        file_prefix = csv_path.stem.split("_")[0]
        
        # Add prefix to all columns except 'date'
        df = df.rename(columns={col: f"{file_prefix}_{col.replace(' ', '_')}" for col in df.columns if col != "date"})
        all_dfs.append(df.set_index("date"))

    combined_df = pd.concat(all_dfs, axis=1)
    combined_df = combined_df.sort_index().reset_index()

    out_path = CRUDE_DATASETS_DIR / "CRUDE.csv"
    combined_df.to_csv(out_path, index=False)
    print(f"CRUDE dataset created at: {out_path}")


if __name__ == "__main__":
    latest_dataset_path_obj = get_latest_crude_dataset_xslx()
    # print(f"Processing: {latest_dataset_path_obj}")
    process_workbook(
        excel_path=latest_dataset_path_obj,
        header_row=4, # row delimiter
        search_start_col=3, # col delimiter
        out_dir=CRUDE_DATASETS_DIR,
        ignore_sheets=set(["META", "GOOG"])
    )

    create_crude()
