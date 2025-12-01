import pandas as pd
from pathlib import Path

from src.constants import CRUDE_DATASETS_DIR


class BloombergExcelParser:
    """Parse Bloomberg Excel datasets into clean CSV format."""
    
    def __init__(self, header_row: int = 4, search_start_col: int = 3):
        """
        Initialize the parser.
        
        Args:
            header_row: Row index where headers are located
            search_start_col: Column index to start searching for data
        """
        self.header_row = header_row
        self.search_start_col = search_start_col
    
    @staticmethod
    def excel_serial_to_date(series):
        """Convert Excel serial date to pandas Timestamp."""
        return pd.to_datetime(series, unit="D", origin="1899-12-30")
    
    @staticmethod
    def is_excel_date_column(col, min_non_na: int = 5) -> bool:
        """Heuristic to detect Excel serial date columns."""
        col = pd.to_numeric(col, errors="coerce")
        non_na = col.notna().sum()
        if non_na < min_non_na:
            return False
        s = col.dropna()
        return ((s > 30000) & (s < 60000)).mean() > 0.8
    
    @staticmethod
    def is_numeric_column(col, min_non_na: int = 5) -> bool:
        """Check if column is numeric with sufficient non-NA values."""
        col = pd.to_numeric(col, errors="coerce")
        return col.notna().sum() >= min_non_na
    
    def parse_indicator_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse a DataFrame to extract time series data organized in pairs of columns: [date, value].
        """
        n_cols = df.shape[1]
        data_start = self.header_row + 1
        all_series = []

        c = self.search_start_col
        while c < n_cols - 1:
            field_code = df.iat[self.header_row, c]

            if pd.isna(field_code):
                c += 1
                continue

            date_col = df.iloc[data_start:, c]
            val_col = df.iloc[data_start:, c + 1]

            if not (self.is_excel_date_column(date_col) and self.is_numeric_column(val_col)):
                c += 1
                continue

            mask = date_col.notna() & val_col.notna()
            if not mask.any():
                c += 2
                continue

            dates = self.excel_serial_to_date(date_col[mask].astype(float))
            values = pd.to_numeric(val_col[mask], errors="coerce")

            s = pd.Series(values.to_list(), index=dates, name=str(field_code))
            all_series.append(s)

            c += 2

        if not all_series:
            return pd.DataFrame()

        out = pd.concat(all_series, axis=1)
        out = out.sort_index().reset_index().rename(columns={"index": "date"})
        return out
    
    def process_workbook(self, excel_path: Path, out_dir: Path, ignore_sheets: set[str] = set()) -> None:
        """
        Process an Excel workbook, extracting tables from each sheet and saving them as CSV files.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        xls = pd.ExcelFile(excel_path)

        for sheet in xls.sheet_names:
            if sheet.split("_")[0] in ignore_sheets or sheet.split("_")[1] == "BLOOMBERG":  # type: ignore
                print(f"[INFO] Ignoring sheet: {sheet}")
                continue
            
            df = xls.parse(sheet, header=None)
            table = self.parse_indicator_block(df)

            if table.empty:
                print(f"[WARNING] Sheet '{sheet}' does not match the pattern.")
                continue

            base = sheet.replace(" ", "_").replace("_CRUDE", "")  # type: ignore
            csv_path = out_dir / f"{base}.csv"
            table.to_csv(csv_path, index=False)
            print(f"Saving: {csv_path}")


class DatasetCombiner:
    """Combine multiple CSV files into a single dataset."""
    
    def __init__(self, datasets_dir: Path = CRUDE_DATASETS_DIR):
        """
        Initialize the combiner.
        
        Args:
            datasets_dir: Directory containing CSV files
        """
        self.datasets_dir = datasets_dir
    
    def get_latest_excel(self) -> Path:
        """Get the latest Excel dataset from the crude datasets directory."""
        all_files = list(self.datasets_dir.glob("*.xlsx"))

        if not all_files:
            raise FileNotFoundError(f"Unable to find files in {self.datasets_dir} that match *.xlsx")

        all_files.sort(reverse=True)
        return all_files[0]
    
    def combine_csvs(self, output_name: str = "CRUDE") -> Path:
        """
        Combine all CSV files into a single dataset with prefixed columns.
        
        Args:
            output_name: Name for the combined CSV file
        
        Returns:
            Path to the created combined CSV
        """
        all_csv_files = list(self.datasets_dir.glob("*.csv"))

        if not all_csv_files:
            raise FileNotFoundError(f"Unable to find files in {self.datasets_dir} that match *.csv")

        all_dfs = []
        for csv_path in all_csv_files:
            if csv_path.stem == output_name:
                continue
            
            df = pd.read_csv(csv_path, parse_dates=["date"])
            file_prefix = csv_path.stem.split("_")[0]
            
            df = df.rename(columns={col: f"{file_prefix}_{col.replace(' ', '_')}" for col in df.columns if col != "date"})
            all_dfs.append(df.set_index("date"))

        combined_df = pd.concat(all_dfs, axis=1)
        combined_df = combined_df.sort_index().reset_index()

        out_path = self.datasets_dir / f"{output_name}.csv"
        combined_df.to_csv(out_path, index=False)
        print(f"Combined dataset created at: {out_path}")
        
        return out_path


if __name__ == "__main__":
    combiner = DatasetCombiner()
    latest_excel = combiner.get_latest_excel()
    
    parser = BloombergExcelParser(header_row=4, search_start_col=3)
    parser.process_workbook(
        excel_path=latest_excel,
        out_dir=CRUDE_DATASETS_DIR,
        ignore_sheets={"META", "GOOG"}
    )
    
    combiner.combine_csvs()
