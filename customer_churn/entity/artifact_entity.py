from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    data_file_path: Path

@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    status_file_path: Path

@dataclass(frozen=True)
class DataTransformationArtifact:
    preprocessor_obj_file_path: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_file_path: Path

@dataclass(frozen=True)
class ModelEvaluationArtifact:
    is_model_accepted: bool
    metric_file_path: Path
