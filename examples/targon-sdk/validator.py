import json
import time
import targon
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

AFFINE_DIR = Path(__file__).resolve().parent
REQUIREMENTS = AFFINE_DIR / "requirements.txt"

image = (
    targon.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install_from_requirements(str(REQUIREMENTS))
    .run_commands(
        [
            # Install Docker (for DIND) using official convenience script
            "curl -fsSL https://get.docker.com -o get-docker.sh",
            "sh get-docker.sh",
            # Install affinetes from GitHub
            "pip install git+https://github.com/affinefoundation/affinetes.git",
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = targon.App("affine", image=image)


@app.function(
    resource=targon.Compute.H200_SMALL, max_replicas=1
)
def sample(
    model_name: str,
    image: str,
    *,
    task_id_start: int = 1,
    task_id_end: int = 10,
    task_ids: list[int] = None,
    llm_port: int = 30000,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """
    Targon serverless function which accepts model M and environment E
    
    This function can be queried as: sample(M, E, task_id_start, task_id_end) -> List[Dict]
    Returns samples from environment E using model M with affinetes + sglang
    
    Args:
        model_name: Model name (M) for evaluation
        image: Affinetes environment image (E) to use
        task_id_start: Start of task ID range (inclusive, default: 1)
        task_id_end: End of task ID range (inclusive, default: 10)
        task_ids: List of specific task IDs (optional, overrides range if provided)
        llm_port: Port for LLM service (vLLM or SGLang)
        timeout: Timeout per task in seconds
        
    Returns:
        Dictionary containing evaluation results
        
    Examples:
        # Sample tasks from range 1-100
        sample("Qwen/Qwen2.5-7B-Instruct", "docker.io/affinefoundation/mth:pi",
               task_id_start=1, task_id_end=100)
        
        # Sample specific task IDs
        sample("Qwen/Qwen2.5-7B-Instruct", "docker.io/affinefoundation/mth:pi",
               task_ids=[1, 5, 10, 15, 20])
    """
    import subprocess
    import asyncio
    import random
    import affinetes as af

    # Determine task IDs based on input parameters
    if task_ids is not None:
        # Use explicit task IDs if provided
        final_task_ids = task_ids
    else:
        # Generate task IDs from range
        final_task_ids = list(range(task_id_start, task_id_end + 1))
    
    print(f"Selected {len(final_task_ids)} task IDs: {final_task_ids[:10]}{'...' if len(final_task_ids) > 10 else ''}")

    # Start Docker daemon (DIND)
    print("Starting Docker daemon...")
    docker_daemon = subprocess.Popen(
        ["dockerd", "--host=unix:///var/run/docker.sock"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Docker to be ready
    time.sleep(10)

    # Start SGLang service
    print(f"Starting SGLang service on port {llm_port}...")
    sglang_process = subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_name,
            "--port",
            str(llm_port),
            "--host",
            "0.0.0.0",
            "--chat-template",
            "llama-2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print("Waiting for LLM service to be ready...")
    time.sleep(60)

    try:
        # Construct base URL for SGLang service
        llm_base_url = f"http://localhost:{llm_port}/v1"
        print(f"base URL: {llm_base_url}")

        # Load affinetes environment with host network mode
        # This allows the container to access sglang service on host network
        print(f"Loading environment from affinetes...")

        async def run_evaluation():
            env = af.load_env(
                image=image,
                host_network=True,
            )

            async def evaluate_task(task_id: int):
                print(f"[Task {task_id}] Starting evaluation...")
                start = time.time()

                try:
                    result = await env.evaluate(
                        model=model_name,
                        base_url=llm_base_url,
                        task_id=task_id,
                        timeout=timeout,
                        _timeout=timeout + 60,
                    )

                    elapsed = time.time() - start
                    print(
                        f"[Task {task_id}] Completed in {elapsed:.2f}s - Score: {result.get('score', 0)}"
                    )

                    return {"task_id": task_id, "result": result, "elapsed": elapsed}
                except Exception as e:
                    print(f"[Task {task_id}] ERROR: {type(e).__name__}: {str(e)}")
                    return {
                        "task_id": task_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

            results = await asyncio.gather(
                *[evaluate_task(task_id) for task_id in final_task_ids]
            )
            return list(results)

        # Run evaluation
        results = asyncio.run(run_evaluation())

        # Calculate summary statistics
        successful_tasks = [r for r in results if "result" in r]
        total_score = sum(r["result"].get("score", 0) for r in successful_tasks)
        avg_score = total_score / len(successful_tasks) if successful_tasks else 0

        return {
            "model_name": model_name,
            "environment_image": image,
            "task_id_range": {"start": task_id_start, "end": task_id_end} if task_id_start else None,
            "total_tasks": len(final_task_ids),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(final_task_ids) - len(successful_tasks),
            "average_score": avg_score,
            "total_score": total_score,
            "results": results,
        }

    finally:
        # Stop SGLang service
        print("Stopping SGLang service...")
        sglang_process.terminate()
        try:
            sglang_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sglang_process.kill()

        # Stop Docker daemon
        print("Stopping Docker daemon...")
        docker_daemon.terminate()
        docker_daemon.wait(timeout=10)


@app.local_entrypoint()
async def main(
    model_name: str,
    image: str = "docker.io/affinefoundation/mth:pi",
    task_id_start: int = 1,
    task_id_end: int = 10,
    task_ids: str = None,
    llm_port: int = 30000,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """
    Run the serverless sample function remotely
    
    Usage:
        # Sample tasks from range 1-100
        targon run examples/targon-sdk/validator.py \
            --model-name "Qwen/Qwen2.5-7B-Instruct" \
            --image "docker.io/affinefoundation/mth:pi" \
            --task-id-start 1 \
            --task-id-end 100
        
        # Sample specific task IDs
        targon run examples/targon-sdk/validator.py \
            --model-name "Qwen/Qwen2.5-7B-Instruct" \
            --image "docker.io/affinefoundation/mth:pi" \
            --task-ids "1,5,10,15,20"
    
    Args:
        model_name: Model name for evaluation
        image: Affinetes environment image to use
        task_id_start: Start of task ID range (inclusive, default: 1)
        task_id_end: End of task ID range (inclusive, default: 10)
        task_ids: Comma-separated task IDs (e.g., "1,5,10,15,20", optional)
        llm_port: Port for LLM service (default: 30000)
        timeout: Timeout per task in seconds
    """
    # Parse task IDs if provided as string
    task_id_list = None
    if task_ids:
        task_id_list = [int(x.strip()) for x in task_ids.split(",")]

    print(f"Starting evaluation:")
    print(f"  Model: {model_name}")
    print(f"  Image: {image}")
    if task_id_list:
        print(f"  Task IDs: {task_id_list}")
    else:
        print(f"  Task ID Range: {task_id_start} - {task_id_end}")
    print(f"  LLM Port: {llm_port}")
    print(f"  Timeout: {timeout}s")
    print()

    result = await sample.remote(
        model_name=model_name,
        image=image,
        task_id_start=task_id_start,
        task_id_end=task_id_end,
        task_ids=task_id_list,
        llm_port=llm_port,
        timeout=timeout,
    )

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"rollouts_{timestamp}.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print summary
    summary = {
        "model_name": result["model_name"],
        "total_tasks": result["total_tasks"],
        "successful_tasks": result["successful_tasks"],
        "failed_tasks": result["failed_tasks"],
        "average_score": result["average_score"],
        "total_score": result["total_score"],
        "output_path": str(output_path),
    }

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    return result
