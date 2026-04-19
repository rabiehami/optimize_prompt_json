"""Database models for optimization run tracking."""

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Run(Base):
    """Stores metadata and final results for an optimization run."""

    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    llm_model = Column(String)
    llm_text_gen_model = Column(String)
    llm_optimizer_model = Column(String)
    batch_size = Column(Integer)
    max_steps = Column(Integer)
    min_steps = Column(Integer)
    num_steps = Column(Integer)
    llm_temp_json_generation = Column(Float)
    llm_temp_text_generation = Column(Float)
    llm_temp_json_extraction = Column(Float)
    json_schema = Column(Text)
    text_path = Column(String, nullable=True)
    schema_path = Column(String, nullable=True)
    created = Column(DateTime(timezone=True), server_default=func.now())

    # Config: targets & thresholds
    field_overlap_target = Column(Float)
    json_distance_target = Column(Float)
    schema_valid_target = Column(Float)
    rollback_threshold = Column(Float)
    evaluate_only = Column(Integer)  # boolean as 0/1
    text_gen_max_tokens = Column(Integer)
    max_tokens = Column(Integer)
    blacklist_fields = Column(Text, nullable=True)  # JSON list
    initial_prompt = Column(Text, nullable=True)

    # Totals
    total_prompt_tokens = Column(Integer)
    total_completion_tokens = Column(Integer)
    total_price = Column(Float)
    total_runtime_seconds = Column(Float)

    # Baseline metrics (step 0)
    baseline_field_overlap = Column(Float)
    baseline_value_similarity = Column(Float)
    baseline_json_distance = Column(Float)
    baseline_schema_valid_rate = Column(Float)
    baseline_score = Column(Float)

    # Final metrics (last step)
    final_field_overlap = Column(Float)
    final_value_similarity = Column(Float)
    final_json_distance = Column(Float)
    final_schema_valid_rate = Column(Float)
    final_score = Column(Float)

    # Prompts & outputs
    optimized_prompt = Column(Text, nullable=True)
    baseline_json = Column(Text, nullable=True)
    optimized_json = Column(Text, nullable=True)


class LLMResponse(Base):
    """Stores individual LLM API requests and responses."""

    __tablename__ = "llm_responses"

    request_id = Column(String, primary_key=True, index=True)
    parent_request_id = Column(String, index=True, nullable=True)
    artifact_id = Column(String, index=True)
    group_id = Column(String, index=True)
    step_id = Column(Integer, index=True)
    run_id = Column(String, index=True)
    llm_model = Column(String)
    llm_temperature = Column(Float)
    json_schema = Column(Text)
    prompt = Column(Text)
    prompt_type = Column(String)
    content = Column(Text)
    json = Column(Text)
    finish_reason = Column(String, nullable=True)
    completion_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    total_tokens = Column(Integer)
    price = Column(Float)
    created = Column(DateTime(timezone=True), server_default=func.now())
    updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class RunMetric(Base):
    """Stores quality metrics for each artifact in each step."""

    __tablename__ = "run_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    artifact_id = Column(String, index=True)
    step_id = Column(Integer, index=True)
    price_step = Column(Float)
    price_cumulative = Column(Float)
    field_overlap = Column(Float)
    value_similarity = Column(Float)
    json_distance = Column(Float)
    schema_valid = Column(Integer)
    created = Column(DateTime(timezone=True), server_default=func.now())


class RunStepMetric(Base):
    """Stores aggregated metrics for each step across all artifacts."""

    __tablename__ = "run_step_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    step_id = Column(Integer, index=True)
    batch_size = Column(Integer)
    price_step = Column(Float)
    price_cumulative = Column(Float)
    field_overlap_mean = Column(Float)
    field_overlap_std = Column(Float)
    value_similarity_mean = Column(Float)
    value_similarity_std = Column(Float)
    json_distance_mean = Column(Float)
    json_distance_std = Column(Float)
    schema_valid_rate = Column(Float)
    created = Column(DateTime(timezone=True), server_default=func.now())
