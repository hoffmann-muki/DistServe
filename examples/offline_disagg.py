import argparse
import importlib

from distserve import OfflineLLM, SamplingParams
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline disaggregated inference with configurable parallelism."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or local path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or local path. Defaults to --model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Optional Ray address to connect to (for multi-node runs). Use 'auto' to connect to an existing cluster.",
    )

    parser.add_argument(
        "--context-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the context/prefill stage.",
    )
    parser.add_argument(
        "--context-pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size for the context/prefill stage.",
    )
    parser.add_argument(
        "--decoding-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the decoding stage.",
    )
    parser.add_argument(
        "--decoding-pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size for the decoding stage.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV-cache block size.",
    )
    parser.add_argument(
        "--max-num-blocks-per-req",
        type=int,
        default=1024,
        help="Maximum number of blocks per request.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory used for KV cache.",
    )
    parser.add_argument(
        "--swap-space",
        type=float,
        default=1.0,
        help="CPU swap space in GB.",
    )

    parser.add_argument(
        "--context-max-batch-size",
        type=int,
        default=4,
        help="Max batch size for context scheduling.",
    )
    parser.add_argument(
        "--context-max-tokens-per-batch",
        type=int,
        default=16384,
        help="Max tokens per batch for context scheduling.",
    )
    parser.add_argument(
        "--decoding-max-batch-size",
        type=int,
        default=4,
        help="Max batch size for decoding scheduling.",
    )
    parser.add_argument(
        "--decoding-max-tokens-per-batch",
        type=int,
        default=16384,
        help="Max tokens per batch for decoding scheduling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling threshold.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens per prompt.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=["\n"],
        help="Stop string. Can be repeated.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Prompt text. Can be repeated. If omitted, built-in prompts are used.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.ray_address is not None:
        ray = importlib.import_module("ray")
        ray.init(address=args.ray_address)

    prompts = args.prompts or [
        "Life blooms like a flower. Far away or by the road. Waiting",
        "A quick brown fox",
        "Artificial intelligence is",
        "To be or not to be,",
        "one two three four",
    ]

    llm = OfflineLLM(
        model_config=ModelConfig(
            model=args.model,
            tokenizer=args.tokenizer,
            dtype=args.dtype,
            seed=args.seed,
        ),
        disagg_parallel_config=DisaggParallelConfig(
            context=ParallelConfig(
                tensor_parallel_size=args.context_tensor_parallel_size,
                pipeline_parallel_size=args.context_pipeline_parallel_size,
            ),
            decoding=ParallelConfig(
                tensor_parallel_size=args.decoding_tensor_parallel_size,
                pipeline_parallel_size=args.decoding_pipeline_parallel_size,
            ),
        ),
        cache_config=CacheConfig(
            block_size=args.block_size,
            max_num_blocks_per_req=args.max_num_blocks_per_req,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_swap_space=args.swap_space,
        ),
        context_sched_config=ContextStageSchedConfig(
            policy="fcfs",
            max_batch_size=args.context_max_batch_size,
            max_tokens_per_batch=args.context_max_tokens_per_batch,
        ),
        decoding_sched_config=DecodingStageSchedConfig(
            policy="fcfs",
            max_batch_size=args.decoding_max_batch_size,
            max_tokens_per_batch=args.decoding_max_tokens_per_batch,
        ),
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
    )

    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    for prompt, step_outputs in zip(prompts, outputs):
        generated_text = "".join(step_output.new_token for step_output in step_outputs)
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text}")
        print(f"Tokens: {len(step_outputs)}")
        print()


if __name__ == "__main__":
    main()