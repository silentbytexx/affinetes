import affinetes as af_env
import os
import sys
import asyncio
from dotenv import load_dotenv
import json

load_dotenv(override=True)


async def main():
    print("\n" + "=" * 60)
    print("Affinetes: Async Environment Execution Example")
    print("=" * 60)

    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("\n   ❌ CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        print("   Or create .env file with: CHUTES_API_KEY=your-key")
        sys.exit(1)

    print("\n1. Loading environment from pre-built image 'affine:latest'...")
    # af_env.build_image_from_env(
    #     env_path="environments/affine",
    #     image_tag="affine:v4",
    # )
    
    env = af_env.load_env(
        image="bignickeye/affine:v4",
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key},
        pull=True,
        cleanup=False,
    )
    print("   ✓ Environment loaded (container started with HTTP server)")

    try:
        result = await env.evaluate(
            task_type="abd",
            task_id=20191,
            model="deepseek-ai/DeepSeek-V3.1",
            base_url="https://llm.chutes.ai/v1",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n   ❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())