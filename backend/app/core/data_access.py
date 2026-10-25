"""Data access layer for parameterized DuckDB queries.

Follows Hard Rule #1: Parameterized SQL only, no raw user concatenation.
"""

from datetime import date
from pathlib import Path
from typing import Any, Optional
import duckdb
import pandas as pd
from app.core.semantic_layer import SeriesDef


class DataAccess:
    """Parameterized DuckDB query interface."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=self._db_path, read_only=True)

    def fetch_series(
        self,
        series_def: SeriesDef,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch a time series between start_date and end_date.

        Returns DataFrame with ['date', 'value'] sorted chronologically.
        """
        con = self._connect()
        try:
            # Build parameterized WHERE clause
            conditions = [
                "is_valid = true",
                f"{series_def.date_column} >= $start_date",
                f"{series_def.date_column} <= $end_date",
                f"{series_def.value_column} IS NOT NULL",
            ]
            params: dict[str, Any] = {
                "start_date": str(start_date),
                "end_date": str(end_date),
            }

            for i, (col, val) in enumerate(series_def.filters.items()):
                param_name = f"filter_{i}"
                conditions.append(f"{col} = ${param_name}")
                params[param_name] = val

            where_clause = " AND ".join(conditions)
            query = f"""
                SELECT CAST({series_def.date_column} AS VARCHAR) AS date,
                       CAST({series_def.value_column} AS DOUBLE) AS value
                FROM {series_def.source_table}
                WHERE {where_clause}
                ORDER BY {series_def.date_column} ASC
            """
            df = con.execute(query, params).fetchdf()
            return df
        finally:
            con.close()

    def data_end_date(self, series_def: SeriesDef) -> Optional[date]:
        """Get latest date available in the dataset for this series."""
        con = self._connect()
        try:
            conditions = [
                "is_valid = true",
                f"{series_def.value_column} IS NOT NULL",
            ]
            params: dict[str, Any] = {}
            for i, (col, val) in enumerate(series_def.filters.items()):
                param_name = f"filter_{i}"
                conditions.append(f"{col} = ${param_name}")
                params[param_name] = val

            where_clause = " AND ".join(conditions)
            query = f"""
                SELECT MAX({series_def.date_column})
                FROM {series_def.source_table}
                WHERE {where_clause}
            """
            result = con.execute(query, params).fetchone()
            if result and result[0]:
                val = result[0]
                if isinstance(val, date):
                    return val
                return pd.Timestamp(val).date()
            return None
        finally:
            con.close()

    def data_start_date(self, series_def: SeriesDef) -> Optional[date]:
        """Get earliest date available in the dataset for this series."""
        con = self._connect()
        try:
            conditions = [
                "is_valid = true",
                f"{series_def.value_column} IS NOT NULL",
            ]
            params: dict[str, Any] = {}
            for i, (col, val) in enumerate(series_def.filters.items()):
                param_name = f"filter_{i}"
                conditions.append(f"{col} = ${param_name}")
                params[param_name] = val

            where_clause = " AND ".join(conditions)
            query = f"""
                SELECT MIN({series_def.date_column})
                FROM {series_def.source_table}
                WHERE {where_clause}
            """
            result = con.execute(query, params).fetchone()
            if result and result[0]:
                val = result[0]
                if isinstance(val, date):
                    return val
                return pd.Timestamp(val).date()
            return None
        finally:
            con.close()

    def row_count(
        self,
        series_def: SeriesDef,
        start_date: date,
        end_date: date,
    ) -> int:
        """Count available non-null rows for this series in date range."""
        con = self._connect()
        try:
            conditions = [
                "is_valid = true",
                f"{series_def.date_column} >= $start_date",
                f"{series_def.date_column} <= $end_date",
                f"{series_def.value_column} IS NOT NULL",
            ]
            params: dict[str, Any] = {
                "start_date": str(start_date),
                "end_date": str(end_date),
            }
            for i, (col, val) in enumerate(series_def.filters.items()):
                param_name = f"filter_{i}"
                conditions.append(f"{col} = ${param_name}")
                params[param_name] = val

            where_clause = " AND ".join(conditions)
            query = f"SELECT COUNT(*) FROM {series_def.source_table} WHERE {where_clause}"
            result = con.execute(query, params).fetchone()
            return int(result[0]) if result else 0
        finally:
            con.close()

    def fetch_aggregated(
        self,
        series_def: SeriesDef,
        start_date: date,
        end_date: date,
        group_by: str,
        agg: str = "avg",
    ) -> pd.DataFrame:
        """Fetch aggregated summary grouped by month, year, or state.

        Args:
            series_def: Target series.
            start_date: Range start.
            end_date: Range end.
            group_by: 'month', 'year', or 'state'.
            agg: 'avg', 'min', or 'max'.
        """
        valid_aggs = {"avg": "AVG", "min": "MIN", "max": "MAX"}
        valid_groups = {
            "month": f"STRFTIME({series_def.date_column}, '%Y-%m')",
            "year": f"STRFTIME({series_def.date_column}, '%Y')",
            "state": "state",
        }
        if group_by not in valid_groups:
            raise ValueError(f"Invalid group_by: {group_by}. Allowed: {list(valid_groups.keys())}")
        agg_fn = valid_aggs.get(agg.lower(), "AVG")
        group_expr = valid_groups[group_by]

        con = self._connect()
        try:
            conditions = [
                "is_valid = true",
                f"{series_def.date_column} >= $start_date",
                f"{series_def.date_column} <= $end_date",
                f"{series_def.value_column} IS NOT NULL",
            ]
            params: dict[str, Any] = {
                "start_date": str(start_date),
                "end_date": str(end_date),
            }
            for i, (col, val) in enumerate(series_def.filters.items()):
                param_name = f"filter_{i}"
                conditions.append(f"{col} = ${param_name}")
                params[param_name] = val

            where_clause = " AND ".join(conditions)
            query = f"""
                SELECT {group_expr} AS group_key,
                       ROUND({agg_fn}({series_def.value_column}), 2) AS value,
                       COUNT(*) AS count
                FROM {series_def.source_table}
                WHERE {where_clause}
                GROUP BY {group_expr}
                ORDER BY group_key ASC
            """
            return con.execute(query, params).fetchdf()
        finally:
            con.close()
