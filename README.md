# openpi-client

`openpi-client` is a Python package for policy-inference communication. It provides a synchronous WebSocket client, a matching WebSocket policy server, and MessagePack serialization for NumPy arrays.

The package is responsible for **sending observations, calling a policy, and returning results**. It does not include OpenPI models, model weights, or model-specific preprocessing. The simulation environment and model inference service can run separately; neither side needs the full OpenPI model repository just to use this communication package.

## How it works

A policy inference request follows this flow:

```text
Simulation client                    Model inference service
WebsocketClientPolicy                WebsocketPolicyServer
        │                                      │
        ├──────── observation dict ───────────>│
        │                                      ├─ policy.infer(observation)
        │<──────────── result dict ────────────┤
```

The client establishes a persistent WebSocket connection. When the connection opens, the server sends metadata once. Each call to `infer()` sends a dictionary; the server decodes it, passes it to `policy.infer(observation)`, and sends the policy's result dictionary back.

NumPy arrays are serialized with the package's `msgpack_numpy` codec, preserving their shape and dtype. The protocol does not assign meaning to observation or action fields. Both sides must agree on field names, data formats, and action semantics.

## Requirements

- Python 3.9 or later
- `websockets` 14 or later
- Communication and array dependencies are installed automatically with the package

The inference server must also install the dependencies required by its model, such as the model framework, CUDA components, and checkpoint-loading libraries. `openpi-client` does not install or manage model dependencies.

## Installation

Install from the repository root:

```bash
python -m pip install .
```

For development, use an editable install:

```bash
python -m pip install -e .
```

With uv:

```bash
uv sync
```

Install the development dependency group with:

```bash
uv sync --group dev
```

If this package is included as a submodule in another project, add its package directory to that project's uv workspace or install it in editable mode into the relevant virtual environment. Install the same `openpi-client` version on both the client and server to share the communication implementation. The server does not need the full OpenPI repository.

## Run a policy server

The server accepts any Python object that implements `infer(observation: dict) -> dict`. Here is a minimal example:

```python
# serve.py
import numpy as np

from openpi_client.base_policy import BasePolicy
from openpi_client.websocket_policy_server import WebsocketPolicyServer


class DummyPolicy(BasePolicy):
    def infer(self, obs: dict) -> dict:
        print("Received keys:", sorted(obs))
        return {"actions": np.zeros((8, 7), dtype=np.float32)}


if __name__ == "__main__":
    policy = DummyPolicy()
    server = WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=8000,
        metadata={"policy": "dummy", "chunk_size": 8, "action_dim": 7},
    )
    server.serve_forever()
```

Run it with:

```bash
python serve.py
```

`metadata` is a dictionary sent to the client once when the connection is established. It can advertise information such as a policy name, action chunk length, or action dimension. The server does not automatically validate policy outputs against this metadata.

`WebsocketPolicyServer` can also run inside an existing asyncio application with `await server.run()`. Do not call `serve_forever()` from an already-running event loop because it creates an event loop with `asyncio.run()`.

### Policy interface

A policy must provide:

```python
class MyPolicy:
    def infer(self, observation: dict) -> dict:
        ...
```

The input is the decoded dictionary sent by the client. The output must be a dictionary that MessagePack can encode; it may contain NumPy arrays and common scalar, list, or dictionary values. A typical action response is:

```python
{
    "actions": np.ndarray,  # For example, shape (K, action_dim)
}
```

The server adds a `server_timing` dictionary to each response. `infer_ms` is the policy inference duration. `prev_total_ms` estimates the previous request's total time from receipt to completed send. Policies should not depend on or overwrite these fields.

## Connect a client

```python
import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy


client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
print("Server metadata:", client.get_server_metadata())

observation = {
    "image": np.zeros((128, 128, 3), dtype=np.uint8),
    "state": np.zeros((7,), dtype=np.float32),
    "prompt": "Move the object to the target.",
}
result = client.infer(observation)
actions = result["actions"]
print(actions.shape, actions.dtype)
print(result.get("server_timing"))
```

`infer()` is synchronous: it waits for the server to finish inference and return a result. The client attempts to connect during construction. If the connection is refused, the current implementation retries periodically without a limit, so start the server first. Close the WebSocket connection explicitly when finished:

```python
client.close()
```

## MIKASA-Robo closed-loop evaluation example

A MIKASA-Robo policy adapter must provide `chunk_size` and `forward(obs)`, returning an action chunk with shape `(K, 7)`. The following example wraps the synchronous WebSocket client as a benchmark policy. Adjust camera splitting, instruction fields, and action conversion to match the actual VLA checkpoint's input and output conventions.

```python
import numpy as np
import torch

from openpi_client.websocket_client_policy import WebsocketClientPolicy


class RemoteChunkPolicy:
    def __init__(self, host: str, port: int, chunk_size: int = 8):
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.chunk_size = chunk_size

    def forward(self, obs):
        def as_numpy(value):
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        request = {
            "rgb": as_numpy(obs["rgb"]),
            "proprio": as_numpy(obs["proprio"]),
        }
        result = self.client.infer(request)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (self.chunk_size, 7):
            raise ValueError(f"Unexpected action shape: {actions.shape}")
        return actions
```

Pass this policy to the project's `evaluate_benchmark(...)`. A real VLA will usually also require a task instruction; the benchmark must pass the instruction to the policy or include it in the request. Matching action dimensions do not guarantee matching semantics. The server output must be converted to the coordinate frame, units, delta/absolute convention, and gripper representation expected by the simulation environment.

## Message format and compatibility

- Requests are MessagePack-encoded dictionaries.
- Responses are MessagePack-encoded dictionaries.
- NumPy arrays use the package's `openpi_client.msgpack_numpy` extension format, which carries raw data, dtype, and shape.
- When the server encounters an exception, it sends an error string and closes the connection. The client raises a `RuntimeError` when it receives a text response instead of binary data.
- The WebSocket server provides a `GET /healthz` health-check path.

Other languages or implementations must follow the same WebSocket message sequence, MessagePack encoding, and NumPy-array extension format to interoperate with this client. If request or response fields are customized, both application adapters must use the same schema.

## Limitations and security notes

- **Synchronous inference:** The client sends one request and waits for its response. This is not an asynchronous action stream or RTC.
- **No reset message:** The client's `reset()` is currently a no-op, and the protocol has no remote episode-reset request. Stateful policies must add a reset protocol or ensure their state does not leak between episodes.
- **Unbounded connection retries:** The client keeps retrying when a connection is refused; it currently has no connection timeout or retry limit.
- **Server concurrency:** The server calls synchronous `policy.infer()` directly from an asyncio handler. This is suitable for a single-client closed-loop evaluation. Slow inference or multiple clients can block the server event loop. Higher-concurrency deployments should use an appropriate thread/process executor, inference queue, or scheduler and load-test it.
- **No server-side authentication:** The client can send an API-key header, but the current server does not validate it. Setting a client key does not enable authentication.
- **Network exposure:** The server binds to `0.0.0.0` by default. Use `127.0.0.1` for local development. For remote deployments, restrict access to a trusted network and enforce access controls at the network layer. Do not expose an unprotected server port to the public internet.
- **Images are not compressed:** WebSocket per-message compression is disabled, so arrays are sent as raw data. Measure network and serialization time before increasing image resolution or request frequency; consider application-level compression only after validating its impact.
- **No model adaptation:** The library does not preprocess images, normalize state, build prompts, load checkpoints, or convert action spaces.

## Development and tests

Install development dependencies and run the tests:

```bash
uv sync --group dev
uv run pytest
```

The current tests primarily cover image utilities and MessagePack array serialization. An end-to-end test should also start a Dummy Policy server, send a request containing NumPy arrays, and verify response fields, shapes, dtypes, and connection shutdown.
