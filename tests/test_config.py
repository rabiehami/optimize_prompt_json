import pytest
from optimize_prompt_json import OptimizationConfig

def test_optimization_config_fields():
    config = OptimizationConfig(schema={"type": "object"}, text="sample")
    assert config.schema == {"type": "object"}
    assert config.text == "sample"
