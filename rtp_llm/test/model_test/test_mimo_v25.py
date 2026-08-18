"""End-to-end test for MiMo V2.5 model.

This script tests loading and running inference on the MiMo V2.5 model
(48 layers, 256 MoE experts, FP8 quantized) through the RTP-LLM server.

GPU requirements:
  - TP=1: single GPU (~80GB VRAM, e.g. A100/H100)
  - TP=2: 2x GPUs
  - TP=8: full EP mode (8 shards, each ~40GB)

Usage (standalone, TP=1 by default):
    python -m rtp_llm.test.model_test.test_mimo_v25

Usage (TP=2):
    TP_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 python -m rtp_llm.test.model_test.test_mimo_v25

Usage (with explicit checkpoint path):
    CHECKPOINT_PATH=/path/to/MiMo-V2.5 python -m rtp_llm.test.model_test.test_mimo_v25
"""

import json
import logging
import os
import sys
import time
import unittest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
_CANDIDATE_PATHS = [
    os.environ.get("CHECKPOINT_PATH", ""),
    "/data1/renkun.ren/models/MiMo-V2.5",
    "/home/renkun.ren/models/MiMo-V2.5",
]


def _find_checkpoint() -> str:
    """Return the first existing checkpoint path, or raise."""
    for p in _CANDIDATE_PATHS:
        if p and os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
            return p
    raise FileNotFoundError(
        "MiMo V2.5 checkpoint not found. Searched:\n  "
        + "\n  ".join(p for p in _CANDIDATE_PATHS if p)
        + "\nSet CHECKPOINT_PATH env var to the correct location."
    )


# ---------------------------------------------------------------------------
# Test class: uses MagaServerManager to start RTP-LLM server, then sends
# an HTTP inference request and verifies the response is non-empty.
# ---------------------------------------------------------------------------
class TestMiMoV25E2E(unittest.TestCase):
    """End-to-end smoke test for MiMo V2.5 via RTP-LLM server."""

    _server = None
    _ckpt_path = None

    @classmethod
    def setUpClass(cls):
        cls._ckpt_path = _find_checkpoint()
        logging.info(f"Using checkpoint: {cls._ckpt_path}")

        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        # Server environment: enable EP mode (default for MoE with tp_size=8)
        env_args = {
            # Suppress real-warmup to speed up startup for tests
            "DSV4_STARTUP_REAL_WARMUP": "0",
        }

        # CLI args forwarded to `python -m rtp_llm.start_server`
        # tp_size: configurable via TP_SIZE env var (default 1 for quick single-GPU test)
        # max_seq_len kept small to reduce KV cache memory for testing
        tp_size = int(os.environ.get("TP_SIZE", "1"))
        smoke_args = (
            f"--model_type mimo_v25 "
            f"--checkpoint_path {cls._ckpt_path} "
            f"--tokenizer_path {cls._ckpt_path} "
            f"--tp_size {tp_size} "
            f"--world_size {tp_size} "
            f"--max_seq_len 2048 "
            f"--concurrency_limit 1"
        )

        cls._server = MagaServerManager(
            env_args=env_args,
            process_file_name="test_mimo_v25.log",
            smoke_args_str=smoke_args,
        )

        logging.info("Starting MiMo V2.5 server (this may take several minutes)...")
        started = cls._server.start_server(timeout=1600)
        if not started:
            cls._server.print_process_log()
            raise RuntimeError(
                "MiMo V2.5 server failed to start. "
                "Check test_mimo_v25.log for details."
            )
        logging.info(f"Server started on port {cls._server.port}")

    @classmethod
    def tearDownClass(cls):
        if cls._server is not None:
            logging.info("Stopping server...")
            cls._server.stop_server()

    # ------------------------------------------------------------------
    # Test: basic text generation
    # ------------------------------------------------------------------
    def test_text_generation_basic(self):
        """Send a simple prompt and verify non-empty generation output."""
        query = {
            "prompt": "The capital of France is",
            "generate_config": {
                "max_new_tokens": 32,
                "top_k": 1,
                "temperature": 0.0,
            },
        }

        success, response_text = self._server.visit(
            query=query,
            retry_times=3,
            endpoint="/",
        )

        self.assertTrue(success, "Server request failed after retries")
        self.assertIsNotNone(response_text, "Response is None")

        # Parse response
        if isinstance(response_text, list):
            # streaming: join chunks
            response_text = b"".join(response_text).decode("utf-8", errors="replace")

        logging.info(f"Raw response: {response_text[:500]}")
        resp = json.loads(response_text)

        # The response structure depends on the endpoint.
        # Typical format: {"response": "...", "finished": true, ...}
        generated = resp.get("response", "")
        logging.info(f"Generated text: {generated}")

        self.assertIsInstance(generated, str)
        self.assertTrue(
            len(generated.strip()) > 0,
            f"Generated text is empty. Full response: {resp}",
        )
        # Basic sanity: the answer should mention "Paris" for this prompt
        # (greedy decoding with temperature=0 should be deterministic)
        logging.info("[PASS] MiMo V2.5 text generation produced non-empty output.")

    # ------------------------------------------------------------------
    # Test: OpenAI-compatible chat endpoint
    # ------------------------------------------------------------------
    def test_openai_chat_completion(self):
        """Send a chat completion request via /v1/chat/completions."""
        query = {
            "model": "mimo_v25",
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer briefly."},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }

        success, response_text = self._server.visit(
            query=query,
            retry_times=3,
            endpoint="/v1/chat/completions",
        )

        self.assertTrue(success, "OpenAI chat request failed after retries")
        self.assertIsNotNone(response_text, "Response is None")

        if isinstance(response_text, list):
            response_text = b"".join(response_text).decode("utf-8", errors="replace")

        logging.info(f"Chat response: {response_text[:500]}")
        resp = json.loads(response_text)

        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        choices = resp.get("choices", [])
        self.assertTrue(len(choices) > 0, f"No choices in response: {resp}")

        content = choices[0].get("message", {}).get("content", "")
        logging.info(f"Chat content: {content}")
        self.assertTrue(
            len(content.strip()) > 0,
            f"Chat content is empty. Full response: {resp}",
        )
        logging.info("[PASS] MiMo V2.5 OpenAI chat endpoint produced valid output.")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow running without bazel: just `python -m rtp_llm.test.model_test.test_mimo_v25`
    try:
        ckpt = _find_checkpoint()
        logging.info(f"Checkpoint found: {ckpt}")
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    unittest.main(verbosity=2)
