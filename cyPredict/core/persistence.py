"""CSV persistence helpers for legacy cyPredict workflows."""

from datetime import datetime
import os
import re

import pandas as pd


class PersistenceMixin:
    """Persist run outputs and reload the most recent optimization settings."""

    def save_dataframe(
        self,
        dataframe,
        folder_path,
        file_name,
        update_column=False,
        update_column_name=None,
        update_column_value=None,
        filter_column_name=None,
        filter_column_value=None
    ):
        """Append a dataframe to a CSV file and optionally update one column.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            New rows to persist. The dataframe index is reset in place before
            appending, matching the legacy behavior.
        folder_path : str or path-like
            Destination folder, interpreted relative to the current working
            directory after sanitizing unsupported characters. Existing
            notebooks often pass a project-relative folder name.
        file_name : str
            CSV file stem. The method appends ``.csv`` after sanitizing the
            value; pass the name without extension to avoid duplicate suffixes.
        update_column : bool, default False
            When ``True``, writes ``update_column_value`` into
            ``update_column_name`` after the existing and new data are
            concatenated.
        update_column_name : str, optional
            Column to create or update. Meaningful only when ``update_column``
            is ``True`` and ``update_column_value`` is not ``None``.
        update_column_value : object, optional
            Value assigned to ``update_column_name``. A ``None`` value disables
            the update branch, even when ``update_column`` is ``True``.
        filter_column_name : str, optional
            Column used to restrict the update to matching rows. It is ignored
            unless both filter parameters are provided.
        filter_column_value : object, optional
            Value matched in ``filter_column_name``. If either filter parameter
            is missing, the update is applied to all rows.

        Returns
        -------
        pandas.DataFrame
            The dataframe written to disk, including any rows previously stored
            in the target CSV.

        Example
        -------
        >>> history = cp.save_dataframe(
        ...     results,
        ...     "runs",
        ...     "QQQ_hp_filter",
        ...     update_column=True,
        ...     update_column_name="run_end_datetime",
        ...     update_column_value="2026-04-29 10:00:00",
        ...     filter_column_name="run_start_datetime",
        ...     filter_column_value="2026-04-29 09:58:00",
        ... )
        """
        clean_folder_name = re.sub(r'[^a-zA-Z0-9_\-\s\:\\]', '', folder_path)
        clean_file_name = re.sub(r'[^a-zA-Z0-9_\-\s\=]', '', file_name)

        folder_path = os.path.join(os.getcwd(), clean_folder_name)
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"{clean_file_name}.csv")

        if os.path.isfile(file_path):
            existing_dataframe = pd.read_csv(file_path)

            existing_dataframe.reset_index(drop=True, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)

            combined_dataframe = pd.concat([existing_dataframe, dataframe], ignore_index=True)
        else:
            combined_dataframe = dataframe

        if update_column == True and update_column_name != None and update_column_value != None:
            if filter_column_name == None and filter_column_value == None:
                combined_dataframe[update_column_name] = update_column_value

            elif filter_column_name != None and filter_column_value != None:
                combined_dataframe.loc[
                    combined_dataframe[filter_column_name] == filter_column_value,
                    update_column_name
                ] = update_column_value

        combined_dataframe.to_csv(file_path, index=False)

        return combined_dataframe

    def get_most_updated_optimization_pars(self, file_path, current_date=None, print_df_code=False):
        """Select optimization parameters closest to an analysis date.

        The method reads the historical optimization CSV produced by the
        notebooks, keeps the legacy HP-filter/non-period-related rows, and
        returns the best row for each ``optimization_label`` using the closest
        available ``analysis_reference_date`` not later than ``current_date``.

        Parameters
        ----------
        file_path : str or path-like
            CSV containing optimization runs. Required columns include
            ``analysis_reference_date``, ``optimization_label``,
            ``opt_period_related_rebuild_range``, ``detrend_type`` and the
            ``best_individual_*`` fields used to build the returned parameter
            table.
        current_date : datetime-like or str, optional
            Reference date used to select historical rows. ``None`` means the
            current system time. String values are parsed by
            ``pandas.to_datetime``.
        print_df_code : bool, default False
            When ``True``, prints Python code that recreates a
            ``cicles_parameters`` dataframe with the selected rows. This is a
            notebook convenience output and does not affect the returned
            dataframe.

        Returns
        -------
        pandas.DataFrame
            Selected rows sorted by ``detrend_type`` and ``min_period``. Legacy
            column names such as ``best_individual_min_period`` are renamed to
            the shorter parameter names expected by downstream notebooks.

        Example
        -------
        >>> pars = cp.get_most_updated_optimization_pars(
        ...     r"D:\\optimizations\\QQQ_runs.csv",
        ...     current_date="2025-12-31",
        ... )
        >>> pars[["min_period", "max_period", "hp_filter_lambda"]]
        """
        df = pd.read_csv(file_path)

        df['analysis_reference_date'] = pd.to_datetime(df['analysis_reference_date'])
        df['best_fitness_value_sum'] = df['best_fitness_value_sum'].fillna(1e10)

        if current_date is None:
            current_date = datetime.now()
        elif isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)

        filtered_df = df[(df['opt_period_related_rebuild_range'] == False) & (df['detrend_type'] == 'hp_filter')]

        result_list = []

        for label, group in filtered_df.groupby('optimization_label'):
            group = group[group['analysis_reference_date'] <= current_date]

            if not group.empty:
                group['time_diff'] = (current_date - group['analysis_reference_date']).abs()

                closest_date = group.loc[group['time_diff'].idxmin(), 'analysis_reference_date']
                closest_group = group[group['analysis_reference_date'] == closest_date]

                idx_min_fitness = closest_group.groupby('optimization_label')['best_fitness_value_sum'].idxmin()

                result = df.loc[idx_min_fitness, [
                    'analysis_reference_date',
                    'optimization_label',
                    'best_individual_min_period',
                    'best_individual_max_period',
                    'detrend_type',
                    'best_fitness_value_sum',
                    'best_individual_hp_filter_lambda',
                    'best_individual_linear_filter_window_size_multiplier',
                    'best_individual_final_kept_n_dominant_circles',
                    'best_individual_num_samples'
                ]]

                result_list.append(result)

        final_result = pd.DataFrame()
        if result_list:
            final_result = pd.concat(result_list).sort_values(by=['detrend_type', 'best_individual_min_period'])

            final_result = final_result.rename(columns={
                'best_individual_min_period': 'min_period',
                'best_individual_max_period': 'max_period',
                'best_individual_hp_filter_lambda': 'hp_filter_lambda',
                'best_individual_final_kept_n_dominant_circles': 'final_kept_n_dominant_circles',
                'best_individual_num_samples': 'num_samples'
            })

            final_result.reset_index(inplace=True)

            if print_df_code:
                result_string = "cicles_parameters = pd.DataFrame(columns = ['num_samples', 'final_kept_n_dominant_circles', 'min_period', 'max_period', 'hp_filter_lambda'])"

                for _, row in final_result.iterrows():
                    result_string += (
                        f"\ncicles_parameters.loc[len(cicles_parameters)] = ["
                        f"{row['num_samples']}, "
                        f"{row['final_kept_n_dominant_circles']}, "
                        f"{row.get('min_period', 'None')}, "
                        f"{row.get('max_period', 'None')}, "
                        f"{int(row['hp_filter_lambda'])}]"
                    )

                print(f"# Hyperparameers for {df['ticker_symbol'].unique()[0]} ticker:")
                print("# -----------------------------------------------------------")
                print(result_string)

        else:
            print("No results for past dates.")

        print(f"analysis_reference_date: {final_result.iloc[0]['analysis_reference_date']}")

        return final_result
