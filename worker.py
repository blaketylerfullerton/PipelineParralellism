import argparse
import pickle
import time
from typing import Optional

import torch
import zmq

from dashboard import get_dashboard, init_dashboard
from model import get_stage, get_tokenizer
from utils import (
    bytes_to_tensor,
    get_forward_port,
    get_telemetry_port,
    get_token_return_port,
    load_config,
    make_activation_msg,
    make_control_msg,
    make_pub_socket,
    make_pull_socket,
    make_push_socket,
    recv_msg,
    send_msg,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline parallelism worker")
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--host", type=str, required=True, help="This machine's IP")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--prompt", type=str, default=None, help="Text prompt (Stage 0 only)")
    p.add_argument("--dashboard", action="store_true")
    return p.parse_args()


def setup_sockets(ctx: zmq.Context, config: dict, stage_id: int, num_stages: int) -> dict:
    sockets: dict = {}

    # Every stage except Stage 0 needs an inbound PULL for hidden states
    if stage_id > 0:
        sockets["pull"] = make_pull_socket(ctx, get_forward_port(config, stage_id))

    # Every stage except the last needs an outbound PUSH for hidden states
    if stage_id < num_stages - 1:
        next_host = config["network"]["workers"][stage_id + 1]
        next_port = get_forward_port(config, stage_id + 1)
        sockets["push"] = make_push_socket(ctx, next_host, next_port)

    # Token return channel: last stage → Stage 0
    if stage_id == num_stages - 1:
        stage0_host = config["network"]["workers"][0]
        sockets["token_push"] = make_push_socket(ctx, stage0_host, get_token_return_port(config))
    if stage_id == 0:
        sockets["token_pull"] = make_pull_socket(ctx, get_token_return_port(config))

    # Telemetry PUB for dashboard
    sockets["pub"] = make_pub_socket(ctx, get_telemetry_port(config, stage_id))

    return sockets


def _publish(pub, stage_id: int, host: str, step: int, state: str,
             elapsed_ms: float, total_steps: int, last_token: str = "") -> None:
    msg = {
        "msg_type": "telemetry",
        "stage_id": stage_id,
        "host": host,
        "step": step,
        "state": state,
        "elapsed_ms": elapsed_ms,
        "total_steps": total_steps,
        "last_token": last_token,
    }
    pub.send(pickle.dumps(msg), zmq.NOBLOCK)


def generation_loop(
    stage_model: torch.nn.Module,
    sockets: dict,
    config: dict,
    stage_id: int,
    num_stages: int,
    host: str,
    prompt: Optional[str],
) -> None:
    max_new_tokens = config["model"].get("max_new_tokens", 20)
    temperature = config["model"].get("temperature", 1.0)
    dash = get_dashboard()

    def _dash_update(**kwargs):
        if dash:
            dash.update_stage(stage_id, host=host, **kwargs)
            dash.tick()

    # ------------------------------------------------------------------ Stage 0
    if stage_id == 0:
        if prompt is None:
            prompt = input("Enter prompt: ")

        tokenizer = get_tokenizer(config)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated = input_ids.clone()

        print(f"\n[Stage 0] Prompt ({input_ids.shape[1]} tokens): \"{prompt}\"")
        print("[Stage 0] Generating: ", end="", flush=True)
        _dash_update(state="running", current_step=0)

        times = []
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            with torch.no_grad():
                hidden = stage_model(generated)

            send_msg(sockets["push"], make_activation_msg(step, 0, hidden))
            _dash_update(state="forward", current_step=step)
            _publish(sockets["pub"], 0, host, step, "forward", 0, step)

            # Wait for token back from last stage
            msg = None
            while msg is None:
                msg = recv_msg(sockets["token_pull"], timeout_ms=60_000)

            next_token = bytes_to_tensor(msg["tensor"], msg["shape"], msg["dtype"]).long()
            generated = torch.cat([generated, next_token], dim=-1)

            word = tokenizer.decode(next_token[0])
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            print(word, end="", flush=True)

            _dash_update(state="forward", current_step=step + 1, last_token=repr(word))
            _publish(sockets["pub"], 0, host, step, "forward", elapsed, step + 1, word)

        # Signal downstream workers to stop
        send_msg(sockets["push"], make_control_msg("end_of_generation"))

        avg_ms = sum(times) / len(times)
        print(f"\n\n[Stage 0] Done. {max_new_tokens} tokens, avg {avg_ms:.1f} ms/token")
        _dash_update(state="done", current_step=max_new_tokens)
        _publish(sockets["pub"], 0, host, max_new_tokens, "done", avg_ms, max_new_tokens)

    # ------------------------------------------------------------ Last stage
    elif stage_id == num_stages - 1:
        print(f"[Stage {stage_id}] Ready — waiting for hidden states...")
        _dash_update(state="waiting", current_step=0)

        times = []
        step = 0
        while True:
            msg = recv_msg(sockets["pull"], timeout_ms=60_000)
            if msg is None or msg["msg_type"] == "end_of_generation":
                break

            hidden = bytes_to_tensor(msg["tensor"], msg["shape"], msg["dtype"])
            _dash_update(state="forward", current_step=step)
            _publish(sockets["pub"], stage_id, host, step, "forward", 0, step)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = stage_model(hidden)          # (1, seq_len, vocab_size)

            # Sample next token from final position
            last_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            send_msg(sockets["token_push"], make_activation_msg(step, stage_id, next_token.float()))

            print(f"[Stage {stage_id}] Step {step:3d}  shape={tuple(hidden.shape)}  {elapsed:.1f}ms")
            _dash_update(state="forward", current_step=step, elapsed_ms=elapsed)
            _publish(sockets["pub"], stage_id, host, step, "forward", elapsed, step)
            step += 1

        avg_ms = sum(times) / max(len(times), 1)
        print(f"[Stage {stage_id}] Done. {step} steps, avg {avg_ms:.1f} ms/step")
        _dash_update(state="done", current_step=step)
        _publish(sockets["pub"], stage_id, host, step, "done", avg_ms, step)

    # --------------------------------------------------------- Middle stages
    else:
        print(f"[Stage {stage_id}] Ready — waiting for hidden states...")
        _dash_update(state="waiting", current_step=0)

        times = []
        step = 0
        while True:
            msg = recv_msg(sockets["pull"], timeout_ms=60_000)
            if msg is None or msg["msg_type"] == "end_of_generation":
                send_msg(sockets["push"], make_control_msg("end_of_generation"))
                break

            hidden = bytes_to_tensor(msg["tensor"], msg["shape"], msg["dtype"])
            _dash_update(state="forward", current_step=step)
            _publish(sockets["pub"], stage_id, host, step, "forward", 0, step)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = stage_model(hidden)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            send_msg(sockets["push"], make_activation_msg(step, stage_id, out))

            print(f"[Stage {stage_id}] Step {step:3d}  shape={tuple(hidden.shape)} → {tuple(out.shape)}  {elapsed:.1f}ms")
            _dash_update(state="forward", current_step=step, elapsed_ms=elapsed)
            _publish(sockets["pub"], stage_id, host, step, "forward", elapsed, step)
            step += 1

        avg_ms = sum(times) / max(len(times), 1)
        print(f"[Stage {stage_id}] Done. {step} steps, avg {avg_ms:.1f} ms/step")
        _dash_update(state="done", current_step=step)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    num_stages = config["pipeline"]["num_stages"]

    if args.stage >= num_stages:
        raise ValueError(f"--stage {args.stage} out of range (num_stages={num_stages})")

    config["network"]["workers"][args.stage] = args.host

    print(f"[Stage {args.stage}] Loading model slice...")
    stage_model = get_stage(args.stage, num_stages, config)
    stage_model.eval()

    ctx = zmq.Context()
    sockets = setup_sockets(ctx, config, args.stage, num_stages)

    # Give all workers time to bind before Stage 0 starts
    time.sleep(1.5)

    dash = None
    if args.dashboard:
        dash = init_dashboard(config, standalone=False)
        for i in range(num_stages):
            dash.update_stage(i, host=config["network"]["workers"][i], state="idle", current_step=0)

    try:
        generation_loop(
            stage_model=stage_model,
            sockets=sockets,
            config=config,
            stage_id=args.stage,
            num_stages=num_stages,
            host=args.host,
            prompt=args.prompt,
        )
    finally:
        if dash:
            time.sleep(1)
            dash.stop()
        for sock in sockets.values():
            sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
