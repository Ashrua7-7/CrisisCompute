from __future__ import annotations

from satya_env.env import RealEnvironment


def _actions_for_hour(hour: int) -> dict:
    if hour == 0:
        return {
            "data_loader": {
                "action": "run_task",
                "task_id": "load_raw_batch_1",
                "cores_needed": 2,
                "gpu_needed": 0,
                "memory_needed": 4,
                "estimated_duration_min": 60,
            },
            "data_cleaner": {
                "action": "wait",
                "task_id": None,
                "cores_needed": 0,
                "gpu_needed": 0,
                "memory_needed": 0,
                "estimated_duration_min": 0,
            },
            "ml_trainer": {
                "action": "wait",
                "task_id": None,
                "cores_needed": 0,
                "gpu_needed": 0,
                "memory_needed": 0,
                "estimated_duration_min": 0,
            },
        }

    if hour == 1:
        return {
            "data_loader": {
                "action": "run_task",
                "task_id": "load_raw_batch_2",
                "cores_needed": 2,
                "gpu_needed": 0,
                "memory_needed": 4,
                "estimated_duration_min": 60,
            },
            "data_cleaner": {
                "action": "run_task",
                "task_id": "clean_batch_1",
                "cores_needed": 4,
                "gpu_needed": 0,
                "memory_needed": 6,
                "estimated_duration_min": 60,
            },
            "ml_trainer": {
                "action": "wait",
                "task_id": None,
                "cores_needed": 0,
                "gpu_needed": 0,
                "memory_needed": 0,
                "estimated_duration_min": 0,
            },
        }

    if hour == 2:
        return {
            "data_loader": {
                "action": "wait",
                "task_id": None,
                "cores_needed": 0,
                "gpu_needed": 0,
                "memory_needed": 0,
                "estimated_duration_min": 0,
            },
            "data_cleaner": {
                "action": "run_task",
                "task_id": "clean_batch_2",
                "cores_needed": 4,
                "gpu_needed": 0,
                "memory_needed": 6,
                "estimated_duration_min": 60,
            },
            "ml_trainer": {
                "action": "wait",
                "task_id": None,
                "cores_needed": 0,
                "gpu_needed": 0,
                "memory_needed": 0,
                "estimated_duration_min": 0,
            },
        }

    return {
        "data_loader": {
            "action": "wait",
            "task_id": None,
            "cores_needed": 0,
            "gpu_needed": 0,
            "memory_needed": 0,
            "estimated_duration_min": 0,
        },
        "data_cleaner": {
            "action": "wait",
            "task_id": None,
            "cores_needed": 0,
            "gpu_needed": 0,
            "memory_needed": 0,
            "estimated_duration_min": 0,
        },
        "ml_trainer": {
            "action": "run_task",
            "task_id": "train_baseline_model",
            "cores_needed": 2,
            "gpu_needed": 1,
            "memory_needed": 8,
            "estimated_duration_min": 120,
        },
    }


def test_environment_smoke() -> None:
    env = RealEnvironment()
    obs = env.reset()

    assert set(obs.keys()) == {"data_loader", "data_cleaner", "ml_trainer"}

    done = False
    hour = 0
    latest_info = {}
    while not done and hour < 8:
        obs, rewards, done, latest_info = env.step(_actions_for_hour(hour))
        assert len(rewards) == 3
        assert "metrics" in latest_info
        hour += 1

    assert latest_info["metrics"]["completed_tasks"] >= 4


if __name__ == "__main__":
    test_environment_smoke()
    print("smoke test passed")
