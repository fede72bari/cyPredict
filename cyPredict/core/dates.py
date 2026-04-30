"""Datetime index extension helpers for the legacy cycle-analysis engine."""

import numpy as np
import pandas as pd


class DatesMixin:
    """Generate future datetime indexes using the cadence of existing data."""

    def find_next_valid_datetime(self, current_datetime, friday_times, saturday_times, sunday_times, workday_times, timezone=None):
        """Return the next timestamp that matches the historical intraday cadence.

        Parameters
        ----------
        current_datetime : pandas.Timestamp
            Last known timestamp. The search starts from this timestamp and
            moves forward until a valid time slot is found.
        friday_times, saturday_times, sunday_times, workday_times : sequence
            Time-of-day samples observed historically for each day type.
            Empty sequences mean that day type is not tradable/available in
            the source dataset. ``workday_times`` is used for non-Friday,
            non-Saturday and non-Sunday dates.
        timezone : tzinfo or str, optional
            Timezone to localize generated timestamps. When omitted, returned
            timestamps remain timezone-naive.

        Returns
        -------
        pandas.Timestamp
            The next valid timestamp after ``current_datetime`` according to
            the provided day/time samples.
        """
        days = 0

        while True:
            temp_date = current_datetime + pd.DateOffset(days=days)

            if temp_date.weekday() == 4 and len(friday_times) > 0:
                day_times = friday_times
            elif temp_date.weekday() == 5 and len(saturday_times) > 0:
                day_times = saturday_times
            elif temp_date.weekday() == 6 and len(sunday_times) > 0:
                day_times = sunday_times
            elif len(workday_times) > 0:
                day_times = workday_times
            else:
                day_times = []

            if len(day_times) > 0:
                if days == 0:
                    current_time_index = np.searchsorted(day_times, current_datetime.time(), side='right')
                    if current_time_index < len(day_times):
                        next_time = day_times[current_time_index]
                        new_datetime = pd.Timestamp.combine(temp_date, next_time)
                        if timezone:
                            new_datetime = new_datetime.tz_localize(timezone)
                        return new_datetime
                    else:
                        days += 1
                        continue
                else:
                    next_time = day_times[0]
                    new_datetime = pd.Timestamp.combine(temp_date, next_time)
                    if timezone:
                        new_datetime = new_datetime.tz_localize(timezone)
                    return new_datetime

            days += 1

    def datetime_dateset_extend(self, df, extension_periods=10, timeframe=None):
        """Append future empty rows to a dataframe using its observed cadence.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with a datetime-like index. Existing rows and
            columns are preserved.
        extension_periods : int, default 10
            Number of future timestamps to append.
        timeframe : str, optional
            Explicit cadence selector. When omitted, the method estimates the
            cadence from median index deltas. Values treated as daily or higher
            cadence are ``"1d"``, ``"1h"``, ``"1wk"`` and ``"1mo"``;
            other inferred values use the intraday branch.

        Returns
        -------
        pandas.DataFrame
            A copy-like dataframe with ``extension_periods`` new rows filled
            with ``NaN`` and indexed by generated future timestamps.

        Notes
        -----
        The method preserves the legacy side effect of writing the extended
        dataframe to ``C:\\Users\\Federico\\Downloads\\df new dates.csv``.
        """
        timezone = df.index.tz

        if timeframe is None:
            deltas = df.index.to_series().diff().dropna()
            median_delta = deltas.median()

            if median_delta >= pd.Timedelta(days=1):
                timeframe = '1d'
            else:
                timeframe = 'intraday'

            if timeframe in ['1d', '1h', '1wk', '1mo']:
                self.log_debug("Daily-like timeframe detected", function="datetime_dateset_extend", timeframe=timeframe)

                historical_weekdays = set(df.index.weekday)

                last_date = df.index.max().normalize()
                new_indexes = []

                next_date = last_date + pd.Timedelta(days=1)

                while len(new_indexes) < extension_periods:
                    if next_date.weekday() in historical_weekdays:
                        new_indexes.append(next_date)

                    next_date += pd.Timedelta(days=1)

                if timezone is not None:
                    new_indexes = [pd.Timestamp(d).tz_localize(timezone) for d in new_indexes]

            else:
                self.log_debug("Intraday timeframe detected", function="datetime_dateset_extend", timeframe=timeframe)

                today = df.index.max().date()

                last_friday = (
                    df.loc[(df.index.date < today) & (df.index.weekday == 4)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_friday):
                    friday_times = df.loc[df.index.date == last_friday].index.time
                else:
                    friday_times = []

                last_saturday = (
                    df.loc[(df.index.date < today) & (df.index.weekday == 5)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_saturday):
                    saturday_times = df.loc[df.index.date == last_saturday].index.time
                else:
                    saturday_times = []

                last_sunday = (
                    df.loc[(df.index.date < today) & (df.index.weekday == 6)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_sunday):
                    sunday_times = df.loc[df.index.date == last_sunday].index.time
                else:
                    sunday_times = []

                last_workday = (
                    df.loc[(df.index.date < today) & (df.index.weekday != 4) & (df.index.weekday != 6)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_workday):
                    workday_times = df.loc[df.index.date == last_workday].index.time
                else:
                    workday_times = []

                last_real_timestamp = df.index.max()

                new_indexes = []
                new_datetime = self.find_next_valid_datetime(
                    last_real_timestamp, friday_times, saturday_times, sunday_times, workday_times, timezone
                )

                for _ in range(extension_periods):
                    new_indexes.append(new_datetime)
                    new_datetime = self.find_next_valid_datetime(
                        new_datetime, friday_times, saturday_times, sunday_times, workday_times, timezone
                    )

        new_rows = pd.DataFrame(np.nan, index=new_indexes, columns=df.columns)
        df = pd.concat([df, new_rows])

        df.to_csv('C:\\Users\\Federico\\Downloads\\df new dates.csv')

        return df
