## Prerequisites

### Install Targon SDK

```bash
pip install git+https://github.com/manifold-inc/targon-sdk.git
```

### Setup Targon Credentials

```bash
pip install keyrings.alt

targon setup
```

## Usage

### Deploy the validator function:
```bash
targon deploy examples/targon-sdk/validator.py
```

### Run evaluation:

#### Sample tasks from a range:
```bash
targon run examples/targon-sdk/validator.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --image "docker.io/affinefoundation/mth:pi" \
    --task-id-start 1 \
    --task-id-end 10 \
    --timeout 1800
```

#### Sample specific task IDs:
```bash
targon run examples/targon-sdk/validator.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --image "docker.io/affinefoundation/mth:pi" \
    --task-ids "1,5,10,15,20" \
    --timeout 1800
```

## API

The serverless function follows the signature: `sample(M, E, task_id_start, task_id_end) -> List[Dict]`

- **M** (model_name): Model name for evaluation
- **E** (image): Affinetes environment image to use
- **task_id_start**: Start of task ID range (inclusive, default: 1)
- **task_id_end**: End of task ID range (inclusive, default: 10)

Additional parameters:
- `task_ids`: Explicitly specify which task IDs to use (overrides range)
- `llm_port`: Port for LLM service (default: 30000)
- `timeout`: Timeout per task in seconds (default: 1800)

## Output

The function returns a JSON object with:
- `model_name`: Model name used for evaluation
- `environment_image`: Environment image used
- `task_id_range`: Task ID range used (if applicable)
- `total_tasks`: Total number of tasks evaluated
- `successful_tasks`: Number of successfully completed tasks
- `failed_tasks`: Number of failed tasks
- `average_score`: Average score across all tasks
- `total_score`: Sum of all task scores
- `results`: Detailed results for each task (List[Dict])

Results are also saved to `rollouts_<timestamp>.json`.