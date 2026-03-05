"""
MOOCCubeX dataset loader.

MOOCCubeX is a knowledge-centred repository from XuetangX, one of the
largest MOOC platforms in China. It provides algorithmically extracted
and expert-validated prerequisite relations (F1 = 0.905), concept-course
mappings, concept-video alignments, and temporal interaction logs.

Reference:
    Yu et al. "MOOCCubeX: A large-scale knowledge-centred repository
    for MOOC research." (2020)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MOOCCubeXLoader:
    """Load and preprocess MOOCCubeX dataset for graph construction.

    The dataset directory should contain:
        entities/
            course.json          — course metadata
            concept.json         — concept metadata
            user.json            — user (student) metadata
        relations/
            prerequisite.json    — concept prerequisite pairs (F1 = 0.905)
            course-concept.json  — course-to-concept mappings
            video-concept.json   — video-to-concept alignments
        interactions/
            enroll.json          — student enrolment records
            video_interaction.json — timestamped video watch logs
            exercise.json        — exercise attempt records

    Parameters
    ----------
    data_dir : str
        Path to the MOOCCubeX root directory.
    min_interactions : int
        Retain only students with >= this many interaction events.
    min_concepts : int
        Retain only courses with >= this many linked concepts.
    subject_filter : str or None
        If provided, filter to courses in this subject (e.g. "computer_science").
    """

    ENTITY_FILES = {
        "course": "entities/course.json",
        "concept": "entities/concept.json",
        "user": "entities/user.json",
    }

    RELATION_FILES = {
        "prerequisite": "relations/prerequisite.json",
        "course_concept": "relations/course-concept.json",
        "video_concept": "relations/video-concept.json",
    }

    INTERACTION_FILES = {
        "enroll": "interactions/enroll.json",
        "video": "interactions/video_interaction.json",
        "exercise": "interactions/exercise.json",
    }

    def __init__(
        self,
        data_dir: str,
        min_interactions: int = 20,
        min_concepts: int = 10,
        subject_filter: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.min_interactions = min_interactions
        self.min_concepts = min_concepts
        self.subject_filter = subject_filter

        # Loaded data containers
        self.courses: Optional[pd.DataFrame] = None
        self.concepts: Optional[pd.DataFrame] = None
        self.students: Optional[pd.DataFrame] = None
        self.prerequisites: Optional[pd.DataFrame] = None
        self.course_concepts: Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None

        # ID mappings (string → integer index)
        self.student_id_map: Dict[str, int] = {}
        self.course_id_map: Dict[str, int] = {}
        self.concept_id_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "MOOCCubeXLoader":
        """Load all entity, relation, and interaction files."""
        logger.info("Loading MOOCCubeX from %s", self.data_dir)

        self._load_entities()
        self._load_relations()
        self._load_interactions()
        self._apply_filters()
        self._build_id_maps()

        logger.info(
            "MOOCCubeX loaded: %d students, %d courses, %d concepts, "
            "%d interactions, %d prerequisites",
            len(self.student_id_map),
            len(self.course_id_map),
            len(self.concept_id_map),
            len(self.interactions),
            len(self.prerequisites),
        )
        return self

    def _load_entities(self):
        """Load course, concept, and user entity files."""
        self.courses = self._read_jsonl("course")
        self.concepts = self._read_jsonl("concept")
        self.students = self._read_jsonl("user")

    def _load_relations(self):
        """Load prerequisite and mapping relations."""
        self.prerequisites = self._read_jsonl("prerequisite")
        self.course_concepts = self._read_jsonl("course_concept")

    def _load_interactions(self):
        """Load and merge interaction logs (enrolments, videos, exercises)."""
        enroll = self._read_jsonl("enroll")
        video = self._read_jsonl("video")
        exercise = self._read_jsonl("exercise")

        # Standardise column names
        for df, itype in [(enroll, "enroll"), (video, "video"), (exercise, "exercise")]:
            if df is not None:
                df["interaction_type"] = itype

        frames = [df for df in [enroll, video, exercise] if df is not None and len(df) > 0]
        self.interactions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _read_jsonl(self, key: str) -> Optional[pd.DataFrame]:
        """Read a JSON-lines file from the appropriate file map."""
        for file_map in [self.ENTITY_FILES, self.RELATION_FILES, self.INTERACTION_FILES]:
            if key in file_map:
                path = os.path.join(self.data_dir, file_map[key])
                break
        else:
            logger.warning("Unknown key: %s", key)
            return None

        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            return None

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        logger.info("Loaded %d records from %s", len(records), path)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filters(self):
        """Apply subject, interaction count, and concept count filters.

        Filters applied (per thesis Section IV-B):
          - Subject filter (optional): retain CS subset
          - Students with >= min_interactions events
          - Courses with >= min_concepts linked concepts
        """
        if self.interactions is None or len(self.interactions) == 0:
            logger.warning("No interactions loaded; skipping filters.")
            return

        # Subject filter
        if self.subject_filter and self.courses is not None:
            if "subject" in self.courses.columns:
                cs_courses = set(
                    self.courses.loc[
                        self.courses["subject"].str.contains(
                            self.subject_filter, case=False, na=False
                        ),
                        "id",
                    ]
                )
                self.interactions = self.interactions[
                    self.interactions["course_id"].isin(cs_courses)
                ]
                logger.info("Subject filter → %d interactions", len(self.interactions))

        # Student interaction count filter
        student_col = self._detect_student_col(self.interactions)
        if student_col:
            counts = self.interactions[student_col].value_counts()
            active = set(counts[counts >= self.min_interactions].index)
            self.interactions = self.interactions[
                self.interactions[student_col].isin(active)
            ]
            logger.info(
                "Student filter (>=%d events) → %d students, %d interactions",
                self.min_interactions,
                len(active),
                len(self.interactions),
            )

        # Course concept count filter
        if self.course_concepts is not None and len(self.course_concepts) > 0:
            cc_counts = self.course_concepts.groupby("course_id").size()
            rich_courses = set(cc_counts[cc_counts >= self.min_concepts].index)
            if "course_id" in self.interactions.columns:
                self.interactions = self.interactions[
                    self.interactions["course_id"].isin(rich_courses)
                ]
            self.course_concepts = self.course_concepts[
                self.course_concepts["course_id"].isin(rich_courses)
            ]
            logger.info(
                "Course filter (>=%d concepts) → %d courses",
                self.min_concepts,
                len(rich_courses),
            )

    @staticmethod
    def _detect_student_col(df: pd.DataFrame) -> Optional[str]:
        """Detect the student identifier column."""
        for col in ["user_id", "student_id", "uid"]:
            if col in df.columns:
                return col
        return None

    # ------------------------------------------------------------------
    # ID mapping
    # ------------------------------------------------------------------

    def _build_id_maps(self):
        """Build string → integer index mappings for all node types."""
        student_col = self._detect_student_col(self.interactions)
        if student_col:
            unique_students = sorted(self.interactions[student_col].unique())
            self.student_id_map = {sid: i for i, sid in enumerate(unique_students)}

        if "course_id" in self.interactions.columns:
            unique_courses = sorted(self.interactions["course_id"].unique())
            self.course_id_map = {cid: i for i, cid in enumerate(unique_courses)}

        # Concepts from course-concept mappings and prerequisites
        concept_ids = set()
        if self.course_concepts is not None and "concept_id" in self.course_concepts.columns:
            concept_ids.update(self.course_concepts["concept_id"].unique())
        if self.prerequisites is not None:
            for col in ["source", "target", "from", "to", "concept_a", "concept_b"]:
                if col in self.prerequisites.columns:
                    concept_ids.update(self.prerequisites[col].unique())
        self.concept_id_map = {cid: i for i, cid in enumerate(sorted(concept_ids))}

    # ------------------------------------------------------------------
    # Output accessors
    # ------------------------------------------------------------------

    def get_edge_lists(self) -> Dict[str, np.ndarray]:
        """Return edge lists for each relation type as (2, E) arrays.

        Returns
        -------
        dict with keys:
            "student_enrols_course"       : (2, E1)
            "student_attempts_concept"    : (2, E2)
            "course_contains_concept"     : (2, E3)
            "concept_prerequisite_concept" : (2, E4)
        """
        edges = {}
        student_col = self._detect_student_col(self.interactions)

        # Student → Course (enrolment)
        if student_col and "course_id" in self.interactions.columns:
            enroll_df = self.interactions[
                self.interactions.get("interaction_type", pd.Series(dtype=str)) == "enroll"
            ]
            if len(enroll_df) == 0:
                enroll_df = self.interactions.drop_duplicates(
                    subset=[student_col, "course_id"]
                )
            src = enroll_df[student_col].map(self.student_id_map).dropna().astype(int)
            dst = enroll_df["course_id"].map(self.course_id_map).dropna().astype(int)
            valid = src.index.intersection(dst.index)
            edges["student_enrols_course"] = np.stack(
                [src.loc[valid].values, dst.loc[valid].values]
            )

        # Student → Concept (attempts via exercise / video interactions)
        concept_col = None
        for col in ["concept_id", "kc_id", "knowledge_id"]:
            if col in self.interactions.columns:
                concept_col = col
                break

        if student_col and concept_col:
            attempt_df = self.interactions.dropna(subset=[concept_col])
            src = attempt_df[student_col].map(self.student_id_map).dropna().astype(int)
            dst = attempt_df[concept_col].map(self.concept_id_map).dropna().astype(int)
            valid = src.index.intersection(dst.index)
            edges["student_attempts_concept"] = np.stack(
                [src.loc[valid].values, dst.loc[valid].values]
            )

        # Course → Concept (contains)
        if self.course_concepts is not None and len(self.course_concepts) > 0:
            src = (
                self.course_concepts["course_id"]
                .map(self.course_id_map)
                .dropna()
                .astype(int)
            )
            dst = (
                self.course_concepts["concept_id"]
                .map(self.concept_id_map)
                .dropna()
                .astype(int)
            )
            valid = src.index.intersection(dst.index)
            edges["course_contains_concept"] = np.stack(
                [src.loc[valid].values, dst.loc[valid].values]
            )

        # Concept → Concept (prerequisite)
        if self.prerequisites is not None and len(self.prerequisites) > 0:
            src_col, dst_col = self._detect_prereq_cols(self.prerequisites)
            if src_col and dst_col:
                src = (
                    self.prerequisites[src_col]
                    .map(self.concept_id_map)
                    .dropna()
                    .astype(int)
                )
                dst = (
                    self.prerequisites[dst_col]
                    .map(self.concept_id_map)
                    .dropna()
                    .astype(int)
                )
                valid = src.index.intersection(dst.index)
                edges["concept_prerequisite_concept"] = np.stack(
                    [src.loc[valid].values, dst.loc[valid].values]
                )

        return edges

    @staticmethod
    def _detect_prereq_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Detect source and target columns for prerequisite relations."""
        candidates = [
            ("source", "target"),
            ("from", "to"),
            ("concept_a", "concept_b"),
            ("pre_concept", "post_concept"),
        ]
        for src, dst in candidates:
            if src in df.columns and dst in df.columns:
                return src, dst
        return None, None

    def get_node_counts(self) -> Dict[str, int]:
        """Return number of nodes per type."""
        return {
            "student": len(self.student_id_map),
            "course": len(self.course_id_map),
            "concept": len(self.concept_id_map),
        }

    def get_student_labels(self) -> Optional[np.ndarray]:
        """Return binary outcome labels for students (if available).

        MOOCCubeX does not provide explicit pass/fail labels.
        We derive a proxy label from exercise performance:
        students with >= 60% correct attempts are labelled 1 (pass).
        """
        if self.interactions is None:
            return None

        exercise_df = self.interactions[
            self.interactions.get("interaction_type", pd.Series()) == "exercise"
        ]
        if len(exercise_df) == 0:
            return None

        student_col = self._detect_student_col(exercise_df)
        score_col = None
        for col in ["score", "correct", "result"]:
            if col in exercise_df.columns:
                score_col = col
                break

        if not student_col or not score_col:
            return None

        scores = exercise_df.groupby(student_col)[score_col].mean()
        labels = np.zeros(len(self.student_id_map), dtype=np.int64)

        for sid, idx in self.student_id_map.items():
            if sid in scores.index:
                labels[idx] = 1 if scores[sid] >= 0.6 else 0

        return labels
