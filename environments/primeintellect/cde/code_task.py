"""Code task generator and evaluator using code execution"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
from typing import Callable

import httpx
from datasets import load_dataset
from openai import AsyncOpenAI

sys.path.insert(0, '/app')
from models import Challenge

# We set higher timeouts than default to avoid judge timeout during eval
HTTPX_TIMEOUT = httpx.Timeout(1200)
HTTPX_LIMITS = httpx.Limits(
    max_connections=8192,
    max_keepalive_connections=8192,
)

logger = logging.getLogger("i3_code")
handler = logging.StreamHandler(sys.stderr)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
logger.addHandler(handler)
logger.setLevel(os.environ.get("I3_CODE_LOG_LEVEL", "INFO"))

# Use original simple instruction prompt
INSTRUCTION_PROMPT = "Solve the programming task below in a Python markdown code block."

# Default timeout per test case (seconds)
DEFAULT_TEST_TIMEOUT = 20


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Try to find ```python blocks first
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Fall back to any ``` blocks
    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no code blocks, return the text as-is
    return text.strip()


class CodeTask:
    """Code task generator and evaluator using INTELLECT-3-RL dataset"""
    
    def __init__(
        self,
        dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
        dataset_subset: str = "code",
        dataset_split: str = "train",
        dataset_shuffle: bool = False,
        difficulty_key: str = "avg@8_qwen3_4b_instruct_2507",
        min_avg_reward: float = 0.0,
        max_avg_reward: float = 1.0,
    ):
        """
        Initialize CodeTask with dataset configuration
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_subset: Dataset subset to use
            dataset_split: Dataset split (train/test/validation)
            dataset_shuffle: Whether to shuffle the dataset
            difficulty_key: Key for filtering by difficulty
            min_avg_reward: Minimum average reward filter
            max_avg_reward: Maximum average reward filter
        """
        logger.info(f"Loading dataset: {dataset_name}/{dataset_subset} split={dataset_split}")
        
        # Load and filter dataset
        def process_example(x):
            info = json.loads(x["info"])
            tests = json.loads(info["tests"])
            
            # Encode inputs/outputs as JSON strings (matching original i3-code implementation)
            # This ensures compatibility with the decode logic in evaluate()
            inputs = [json.dumps(tests["inputs"][i]) for i in range(len(tests["inputs"]))]
            outputs = [json.dumps(tests["outputs"][i]) for i in range(len(tests["outputs"]))]
            
            # Rebuild tests dict with encoded strings
            encoded_tests = {
                "inputs": inputs,
                "outputs": outputs
            }
            
            return {
                "question": INSTRUCTION_PROMPT + "\n\n" + x["question"],
                "tests": json.dumps(encoded_tests),  # Store as JSON string
                "source": info.get("source", "")
            }
        
        self.dataset = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            .filter(lambda x: min_avg_reward <= x.get(difficulty_key, 0) <= max_avg_reward)
            .map(process_example)
        )
        
        if dataset_shuffle:
            self.dataset = self.dataset.shuffle(seed=42)
        
        logger.info(f"Dataset loaded: {len(self.dataset)} examples")
    
    async def generate(self, task_id: int = None) -> Challenge:
        """
        Generate a code task challenge
        
        Args:
            task_id: Optional task ID for deterministic selection.
                     If provided, used as index into dataset.
                     If not provided, random sample is selected.
        """
        if task_id is not None:
            # Use task_id as index for deterministic selection
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            # Random selection
            import random
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        return Challenge(
            env="code",
            prompt=sample["question"],
            extra={
                "tests": sample["tests"],
                "source": sample.get("source", ""),
                "task_id": task_id,
                "dataset_index": idx
            }
        )
    
    async def evaluate(self, response: str, challenge: Challenge, timeout: int = DEFAULT_TEST_TIMEOUT) -> tuple[float, str]:
        """
        Evaluate code response by running test cases in parallel using stdin/stdout
        
        Args:
            response: Model response containing code
            challenge: Original challenge with test cases
            timeout: Timeout per test case in seconds
        
        Returns:
            Tuple of (score, test_result):
                - score: 1.0 if all tests pass, else 0.0
                - test_result: String in format "passed/total" (e.g., "7/15")
        """
        tests_str = challenge.extra.get("tests", "")
        if not tests_str:
            logger.warning("No tests provided")
            return 0.0, "0/0"
        
        # Extract code from response
        code = extract_code_from_markdown(response)
        if not code:
            logger.warning("No code found in response")
            return 0.0, "0/0"
        
        logger.debug(f"Extracted code:\n{code}")
        
        # Parse tests
        try:
            tests = json.loads(tests_str)
        except Exception as e:
            logger.error(f"Failed to parse tests: {e}")
            return 0.0, "0/0"
        
        # Extract test data (inputs and outputs are JSON strings)
        inputs = tests.get("inputs", [])
        outputs = tests.get("outputs", [])
        
        if not inputs or not outputs:
            logger.warning("No test inputs/outputs found")
            return 0.0, "0/0"
        
        if len(inputs) != len(outputs):
            logger.error(f"Mismatch: {len(inputs)} inputs vs {len(outputs)} outputs")
            return 0.0, f"0/{len(inputs)}"
        
        # Run tests in parallel using asyncio.gather
        total = len(inputs)
        tasks = []
        
        for i in range(total):
            try:
                # Parse input and output (they are JSON strings)
                test_input_str = json.loads(inputs[i])
                expected_output_str = json.loads(outputs[i])
                
                # Convert to string for stdin - inputs already contain proper formatting
                if isinstance(test_input_str, str):
                    stdin_input = test_input_str
                else:
                    # Should not happen based on task format, but handle gracefully
                    stdin_input = str(test_input_str)
                
                # Create task for parallel execution
                task = self._run_test_with_subprocess(
                    code=code,
                    stdin_input=stdin_input,
                    expected_output=expected_output_str,
                    timeout=timeout,
                    test_index=i
                )
                tasks.append(task)
                    
            except Exception as e:
                logger.debug(f"Test {i}: Failed to prepare - {e}")
                # Add a failed task
                tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Placeholder that returns None
        
        # Execute all tests in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count passed tests
        passed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.debug(f"Test {i}: EXCEPTION - {result}")
            elif result is True:
                passed += 1
                logger.debug(f"Test {i}: PASSED")
            else:
                logger.debug(f"Test {i}: FAILED")
        
        # Calculate score (binary: 1.0 if all pass, else 0.0)
        pass_rate = passed / total if total > 0 else 0.0
        passed_all = (pass_rate == 1.0)
        score = 1.0 if passed_all else 0.0
        
        # Format test result as "passed/total"
        test_result = f"{passed}/{total}"
        
        logger.info(f"Evaluation complete: {test_result} tests passed, pass_rate={pass_rate:.2%}, score={score}")
        
        return score, test_result
    
    async def _run_test_with_subprocess(
        self,
        code: str,
        stdin_input: str,
        expected_output: str,
        timeout: int,
        test_index: int
    ) -> bool:
        """
        Run a single test case in a subprocess with strict timeout and cleanup
        
        Args:
            code: Python code to execute
            stdin_input: Input to provide via stdin
            expected_output: Expected output string
            timeout: Timeout in seconds
            test_index: Index of the test (for logging)
        
        Returns:
            True if test passed, False otherwise
        """
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        process = None
        try:
            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                'python3', temp_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Set resource limits
                preexec_fn=lambda: self._set_process_limits() if hasattr(os, 'setrlimit') else None
            )
            
            try:
                # Run with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_input.encode('utf-8')),
                    timeout=timeout
                )
                
                # Check exit code
                if process.returncode != 0:
                    logger.debug(f"Test {test_index}: Non-zero exit code {process.returncode}")
                    if stderr:
                        logger.debug(f"Test {test_index}: stderr: {stderr.decode('utf-8', errors='ignore')[:200]}")
                    return False
                
                # Compare output
                actual_output = stdout.decode('utf-8', errors='ignore').strip()
                expected_output_stripped = expected_output.strip()
                
                if actual_output == expected_output_stripped:
                    return True
                else:
                    logger.debug(f"Test {test_index}: Output mismatch")
                    logger.debug(f"Expected: {expected_output_stripped[:200]}")
                    logger.debug(f"Got: {actual_output[:200]}")
                    return False
                    
            except asyncio.TimeoutError:
                logger.debug(f"Test {test_index}: Timeout after {timeout}s")
                # Kill process group to ensure all child processes are terminated
                if process and process.returncode is None:
                    try:
                        # Try to kill process group first
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, AttributeError):
                        pass
                    
                    try:
                        # Also try direct kill
                        process.kill()
                        await process.wait()
                    except (ProcessLookupError, PermissionError):
                        pass
                return False
                
        except Exception as e:
            logger.debug(f"Test {test_index}: Exception during execution: {e}")
            return False
        finally:
            # Ensure process is terminated
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except (ProcessLookupError, PermissionError):
                    pass
            
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    @staticmethod
    def _set_process_limits():
        """Set resource limits for subprocess to prevent resource exhaustion"""
        try:
            import resource
            # Limit virtual memory to 10GB
            resource.setrlimit(resource.RLIMIT_AS, (10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
            # Limit CPU time to prevent infinite loops
            resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        except (ImportError, OSError):
            pass