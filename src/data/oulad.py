"""
Open University Learning Analytics Dataset (OULAD) loader.

OULAD contains records from 32,593 students across 22 presentations
of seven modules at the UK Open University during 2013-2014.
Provides demographic attributes (gender, age band, disability, IMD band),
10,655,280 daily VLE click summaries across 20 activity types,
continuous assessment scores, and final outcomes.

Reference:
    Kuzilek et al. "Open University Learning Analytics Dataset."
    Scientific Data, 4:170171 (2017). CC-BY 4.0
"""

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OULADLoader:
    """Load and preprocess the OULAD dataset.

    The dataset directory should contain the CSV files from the
    official OULAD release:
        studentInfo.csv
        studentRegistration.csv
        studentAssessment.csv
        studentVle.csv
        assessments.csv
        courses.csv
        vle.csv

    Parameters
    ----------
    data_dir : str
        Path to the OULAD directory containing CSV files.
    co_occurrence_window : int
        Time window in seconds for constructing co-occurrence edges
        between VLE activities. Default: 86400 (24 hours).
    include_withdrawn : bool
        Whether to include students who withdrew. Default True,
        as withdrawal patterns carry engagement information.
    """

    CSV_FILES = [
        "studentInfo.csv",
        "studentRegistration.csv",
        "studentAssessment.csv",
        "studentVle.csv",
        "assessments.csv",
        "courses.csv",
        "vle.csv",
    ]

    # Final outcome mapping
    OUTCOME_MAP = {
        "Distinction": 1,
        "Pass": 1,
        "Fail": 0,
        "Withdrawn": 0,
    }

    def __init__(
        self,
        data_dir: str,
        co_occurrence_window: int = 86400,
        include_withdrawn: bool = True,
    ):
        self.data_dir = data_dir
        self.co_occurrence_window = co_occurrence_window
        self.include_withdrawn = include_withdrawn

        # Raw tables
        self.student_info: Optional[pd.DataFrame] = None
        self.student_vle: Optional[pd.DataFrame] = None
        self.student_assessment: Optional[pd.DataFrame] = None
        self.vle: Optional[pd.DataFrame] = None
        self.assessments: Optional[pd.DataFrame] = None
        self.courses: Optional[pd.DataFrame] = None

        # ID mappings
        self.student_id_map: Dict[int, int] = {}
        self.module_id_map: Dict[str, int] = {}
        self.activity_id_map: Dict[int, int] = {}   # VLE site_id → index

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "OULADLoader":
        """Load all CSV files and build ID mappings."""
        logger.info("Loading OULAD from %s", self.data_dir)

        self.student_info = self._read_csv("studentInfo.csv")
        self.student_vle = self._read_csv("studentVle.csv")
        self.student_assessment = self._read_csv("studentAssessment.csv")
        self.vle = self._read_csv("vle.csv")
        self.assessments = self._read_csv("assessments.csv")
        self.courses = self._read_csv("courses.csv")

        self._preprocess()
        self._build_id_maps()

        logger.info(
            "OULAD loaded: %d students, %d modules, %d VLE activities, "
            "%d interaction records",
            len(self.student_id_map),
            len(self.module_id_map),
            len(self.activity_id_map),
            len(self.student_vle),
        )
        return self

    def _read_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Read a CSV file from the data directory."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            return None
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), filename)
        return df

    def _preprocess(self):
        """Filter and preprocess loaded tables."""
        if self.student_info is None:
            return

        # Filter withdrawn students if requested
        if not self.include_withdrawn:
            self.student_info = self.student_info[
                self.student_info["final_result"] != "Withdrawn"
            ]
            logger.info(
                "Excluded withdrawn students → %d remaining",
                len(self.student_info),
            )

        # Build unique student key: (id_student, code_module, code_presentation)
        self.student_info["student_key"] = (
            self.student_info["id_student"].astype(str)
            + "_"
            + self.student_info["code_module"]
            + "_"
            + self.student_info["code_presentation"]
        )

        # Merge student key into VLE interactions
        if self.student_vle is not None:
            self.student_vle["student_key"] = (
                self.student_vle["id_student"].astype(str)
                + "_"
                + self.student_vle["code_module"]
                + "_"
                + self.student_vle["code_presentation"]
            )
            # Keep only students present in student_info
            valid_keys = set(self.student_info["student_key"])
            self.student_vle = self.student_vle[
                self.student_vle["student_key"].isin(valid_keys)
            ]

    def _build_id_maps(self):
        """Build integer index mappings for students, modules, and activities."""
        if self.student_info is not None:
            unique_students = sorted(self.student_info["student_key"].unique())
            self.student_id_map = {sk: i for i, sk in enumerate(unique_students)}

        if self.student_info is not None:
            unique_modules = sorted(self.student_info["code_module"].unique())
            self.module_id_map = {m: i for i, m in enumerate(unique_modules)}

        if self.vle is not None:
            unique_sites = sorted(self.vle["id_site"].unique())
            self.activity_id_map = {s: i for i, s in enumerate(unique_sites)}

    # ------------------------------------------------------------------
    # Edge lists
    # ------------------------------------------------------------------

    def get_edge_lists(self) -> Dict[str, np.ndarray]:
        """Return edge lists for the OULAD heterogeneous graph.

        Edge types:
            student_clicks_activity   : student interacted with VLE activity
            module_contains_activity  : VLE activity belongs to module
            activity_co_occurrence    : two activities interacted by same student
                                        within co_occurrence_window

        Returns
        -------
        dict of str → np.ndarray of shape (2, num_edges)
        """
        edges = {}

        # Student → Activity (clicks)
        if self.student_vle is not None:
            pairs = self.student_vle[["student_key", "id_site"]].drop_duplicates()
            src = pairs["student_key"].map(self.student_id_map).dropna().astype(int)
            dst = pairs["id_site"].map(self.activity_id_map).dropna().astype(int)
            valid = src.index.intersection(dst.index)
            if len(valid) > 0:
                edges["student_clicks_activity"] = np.stack(
                    [src.loc[valid].values, dst.loc[valid].values]
                )

        # Module → Activity (contains)
        if self.vle is not None:
            pairs = self.vle[["code_module", "id_site"]].drop_duplicates()
            src = pairs["code_module"].map(self.module_id_map).dropna().astype(int)
            dst = pairs["id_site"].map(self.activity_id_map).dropna().astype(int)
            valid = src.index.intersection(dst.index)
            if len(valid) > 0:
                edges["module_contains_activity"] = np.stack(
                    [src.loc[valid].values, dst.loc[valid].values]
                )

        # Activity ↔ Activity (temporal co-occurrence)
        co_occ = self._build_co_occurrence_edges()
        if co_occ is not None:
            edges["activity_co_occurrence"] = co_occ

        return edges

    def _build_co_occurrence_edges(self) -> Optional[np.ndarray]:
        """Build co-occurrence edges between VLE activities.

        Two activities are linked if the same student interacts with
        both within a 24-hour window (per thesis Section IV-C).
        """
        if self.student_vle is None or len(self.student_vle) == 0:
            return None

        logger.info("Building co-occurrence edges (window=%ds)...", self.co_occurrence_window)

        # Convert date to numeric for windowing
        vle = self.student_vle[["student_key", "id_site", "date"]].copy()
        vle["date"] = pd.to_numeric(vle["date"], errors="coerce")
        vle = vle.dropna(subset=["date"])

        # Sort by student and date
        vle = vle.sort_values(["student_key", "date"])

        # For efficiency, process per student
        window_days = self.co_occurrence_window / 86400  # OULAD dates are in days

        edge_set = set()

        for _, group in vle.groupby("student_key"):
            sites = group["id_site"].values
            dates = group["date"].values

            for i in range(len(sites)):
                for j in range(i + 1, len(sites)):
                    if abs(dates[j] - dates[i]) <= window_days:
                        a = self.activity_id_map.get(sites[i])
                        b = self.activity_id_map.get(sites[j])
                        if a is not None and b is not None and a != b:
                            edge_set.add((min(a, b), max(a, b)))
                    else:
                        break  # sorted by date, no need to check further

        if not edge_set:
            return None

        edges = np.array(list(edge_set), dtype=np.int64).T
        # Make undirected
        edges = np.concatenate([edges, edges[[1, 0]]], axis=1)

        logger.info("Built %d co-occurrence edges", edges.shape[1])
        return edges

    # ------------------------------------------------------------------
    # Labels and demographics
    # ------------------------------------------------------------------

    def get_node_counts(self) -> Dict[str, int]:
        return {
            "student": len(self.student_id_map),
            "module": len(self.module_id_map),
            "activity": len(self.activity_id_map),
        }

    def get_student_labels(self) -> np.ndarray:
        """Return binary pass/fail labels for each student.

        Mapping: Distinction, Pass → 1; Fail, Withdrawn → 0
        """
        labels = np.zeros(len(self.student_id_map), dtype=np.int64)

        if self.student_info is None:
            return labels

        for _, row in self.student_info.iterrows():
            sk = row.get("student_key")
            outcome = row.get("final_result")
            if sk in self.student_id_map and outcome in self.OUTCOME_MAP:
                labels[self.student_id_map[sk]] = self.OUTCOME_MAP[outcome]

        return labels

    def get_demographics(self) -> Optional[pd.DataFrame]:
        """Return demographic attributes for each student.

        Columns: gender, age_band, disability, imd_band, region,
                 highest_education, num_of_prev_attempts.
        """
        if self.student_info is None:
            return None

        demo_cols = [
            "student_key", "gender", "age_band", "disability",
            "imd_band", "region", "highest_education",
            "num_of_prev_attempts",
        ]
        available = [c for c in demo_cols if c in self.student_info.columns]
        return self.student_info[available].copy()

    def get_withdrawn_mask(self) -> np.ndarray:
        """Return boolean mask for students who withdrew before completion.

        Used for Objective 3 analysis: testing engagement signatures
        on withdrawn students (n ≈ 4,218 per thesis Section V-C).
        """
        mask = np.zeros(len(self.student_id_map), dtype=bool)

        if self.student_info is None:
            return mask

        withdrawn = self.student_info[
            self.student_info["final_result"] == "Withdrawn"
        ]
        for _, row in withdrawn.iterrows():
            sk = row.get("student_key")
            if sk in self.student_id_map:
                mask[self.student_id_map[sk]] = True

        return mask

    def get_engagement_features(self) -> Optional[np.ndarray]:
        """Compute traditional engagement metrics per student.

        Features (per thesis Table 5 reference condition):
            - total_clicks: sum of all VLE click counts
            - login_frequency: number of distinct days with activity
            - time_on_task: proxy from total clicks (no duration in OULAD)
            - session_count: number of distinct (day, activity_type) pairs
        """
        if self.student_vle is None:
            return None

        features = np.zeros((len(self.student_id_map), 4), dtype=np.float32)
        grouped = self.student_vle.groupby("student_key")

        for sk, group in grouped:
            if sk not in self.student_id_map:
                continue
            idx = self.student_id_map[sk]
            features[idx, 0] = group["sum_click"].sum()       # total clicks
            features[idx, 1] = group["date"].nunique()         # login frequency
            features[idx, 2] = group["sum_click"].sum()        # time proxy
            features[idx, 3] = len(group)                      # session count

        return features
