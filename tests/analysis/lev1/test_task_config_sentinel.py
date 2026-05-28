"""Dual-task YAMLs load after porting network_glm model definitions."""
from neuro_workflow.analysis.task_config.loader import get_task_parameters, get_task_regressors


def test_dual_task_regressors_load():
    regressors = get_task_regressors("cuedTSWFlanker")
    assert "cue_stay_task_stay_congruent" in regressors
    assert "response_time" in regressors


def test_dual_task_parameters_load():
    # Parameters (tr, dummy_scans, ...) remain available for pipeline planning.
    params = get_task_parameters("cuedTSWFlanker")
    assert params["tr"] == 1.49
    assert params["expected_sessions"] == 2


def test_base_task_loads_normally():
    params = get_task_parameters("stopSignal")
    assert params["tr"] == 1.49
