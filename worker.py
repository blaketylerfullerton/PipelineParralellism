import argparse
import pickle
import random
import time
from typing import Optional

import torch
import zmq

from dashboard import get_dashboard, init_dashboard
from model import get_stage, get_tokenizer
from utils import (
    get_forward_port,
    get_telemetry_port,
    get_token_return_port,
    load_config,
    make_activation_msg,
    make_control_msg,
    make_pub_socket,
    make_pull_socket,
    make_push_socket,
    make_spec_result_msg,
    recv_msg,
    send_msg,
    tensor_from_activation_msg,
    trim_dynamic_cache,
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
    codec = config.get("compression", {}).get("mode", "fp32")
    spec_cfg = config.get("speculative", {})
    spec_enabled = spec_cfg.get("enabled", False)
    k = spec_cfg.get("k", 4)
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

        draft_model = None
        if spec_enabled:
            from draft import DraftModel
            draft_model = DraftModel(spec_cfg.get("draft_model", "gpt2"), temperature=temperature)

        print(f"\n[Stage 0] Prompt ({input_ids.shape[1]} tokens): \"{prompt}\"")
        print("[Stage 0] Generating: ", end="", flush=True)
        _dash_update(state="running", current_step=0)

        times = []
        stage_kv = None
        kv_len = 0
        tokens_generated = 0
        spec_step = 0
        accepted_total = 0
        spec_steps_total = 0

        while tokens_generated < max_new_tokens:
            is_prefill = (spec_step == 0)
            t0 = time.perf_counter()

            if is_prefill:
                # ---- Prefill (always non-speculative) ----
                with torch.no_grad():
                    hidden, stage_kv = stage_model(generated, past_key_values=None)
                kv_len = generated.shape[1]

                send_msg(sockets["push"], make_activation_msg(
                    spec_step, 0, hidden, is_prefill=True, codec=codec))
                _dash_update(state="forward", current_step=0)
                _publish(sockets["pub"], 0, host, spec_step, "forward", 0, 0)

                msg = None
                while msg is None:
                    msg = recv_msg(sockets["token_pull"], timeout_ms=60_000)

                if spec_enabled:
                    next_token = torch.tensor([[msg["target_samples"][0]]], dtype=torch.long)
                    draft_model.init_cache(generated)  # prime draft cache on prompt only
                else:
                    next_token = tensor_from_activation_msg(msg).long()

                generated = torch.cat([generated, next_token], dim=-1)
                word = tokenizer.decode(next_token[0])
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed)
                print(word, end="", flush=True)
                tokens_generated += 1

            elif spec_enabled:
                # ---- Speculative decode ----
                last_tok = generated[:, -1:]  # (1,1) — not yet in KV cache
                kv_before = kv_len

                draft_tokens_t, draft_probs_t = draft_model.draft_k_tokens(last_tok, k)
                # draft_tokens_t: (1, k), draft_probs_t: (k,)

                feed = torch.cat([last_tok, draft_tokens_t], dim=1)  # (1, k+1)
                with torch.no_grad():
                    hidden, stage_kv = stage_model(feed, past_key_values=stage_kv)

                send_msg(sockets["push"], make_activation_msg(
                    spec_step, 0, hidden, is_prefill=False, codec=codec,
                    draft_tokens=draft_tokens_t[0].tolist()))
                _dash_update(state="forward", current_step=tokens_generated)
                _publish(sockets["pub"], 0, host, spec_step, "forward", 0, tokens_generated)

                msg = None
                while msg is None:
                    msg = recv_msg(sockets["token_pull"], timeout_ms=60_000)

                target_probs = msg["target_probs"]    # [k] floats
                target_samples = msg["target_samples"]  # [k+1] ints

                # Speculative accept/reject (Leviathan et al. 2023, Algorithm 1)
                accepted = []
                for i in range(k):
                    p_i = target_probs[i]
                    q_i = draft_probs_t[i].item()
                    if random.random() <= min(1.0, p_i / (q_i + 1e-9)):
                        accepted.append(draft_tokens_t[0, i].item())
                    else:
                        accepted.append(target_samples[i])  # resample from adjusted dist
                        break
                else:
                    accepted.append(target_samples[k])  # bonus token when all k accepted

                # Clip to remaining budget
                remaining = max_new_tokens - tokens_generated
                accepted = accepted[:remaining]
                M = len(accepted)

                # When all K drafts are accepted, draft KV is at kv_before+k (processed
                # seed_token..draft_{k-1}), but target KV is at kv_before+k+1 (also
                # processed draft_k). Advance draft by one to stay in sync.
                if M == k + 1:
                    draft_model.advance_cache(draft_tokens_t[0, k - 1].item())

                # Trim all caches to consistent length
                trim_to = kv_before + M
                trim_dynamic_cache(stage_kv, trim_to)
                draft_model.trim_cache(trim_to)
                kv_len = trim_to
                send_msg(sockets["push"], make_control_msg("trim_cache", keep_len=trim_to))

                elapsed = (time.perf_counter() - t0) * 1000
                per_tok_ms = elapsed / M

                new_tokens_t = torch.tensor([accepted], dtype=torch.long)
                generated = torch.cat([generated, new_tokens_t], dim=1)
                for tok_id in accepted:
                    print(tokenizer.decode([tok_id]), end="", flush=True)
                    times.append(per_tok_ms)
                tokens_generated += M
                accepted_total += M
                spec_steps_total += 1

            else:
                # ---- Non-speculative decode ----
                with torch.no_grad():
                    hidden, stage_kv = stage_model(generated[:, -1:], past_key_values=stage_kv)

                send_msg(sockets["push"], make_activation_msg(
                    spec_step, 0, hidden, is_prefill=False, codec=codec))
                _dash_update(state="forward", current_step=tokens_generated)
                _publish(sockets["pub"], 0, host, spec_step, "forward", 0, tokens_generated)

                msg = None
                while msg is None:
                    msg = recv_msg(sockets["token_pull"], timeout_ms=60_000)

                next_token = tensor_from_activation_msg(msg).long()
                generated = torch.cat([generated, next_token], dim=-1)
                word = tokenizer.decode(next_token[0])
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed)
                print(word, end="", flush=True)
                tokens_generated += 1

            _dash_update(state="forward", current_step=tokens_generated)
            _publish(sockets["pub"], 0, host, spec_step, "forward",
                     times[-1] if times else 0, tokens_generated)
            spec_step += 1

        send_msg(sockets["push"], make_control_msg("end_of_generation"))

        avg_ms = sum(times) / max(len(times), 1)
        tps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"\n\n[Stage 0] Done. {tokens_generated} tokens, avg {avg_ms:.1f} ms/token ({tps:.2f} TPS)")
        if spec_enabled and spec_steps_total > 0:
            avg_acc = accepted_total / spec_steps_total
            print(f"[Stage 0] Spec:  avg {avg_acc:.2f} tokens/step  "
                  f"(k={k}, acceptance={avg_acc / (k + 1):.1%})")
        _dash_update(state="done", current_step=tokens_generated)
        _publish(sockets["pub"], 0, host, spec_step, "done", avg_ms, tokens_generated)

    # ------------------------------------------------------------ Last stage
    elif stage_id == num_stages - 1:
        print(f"[Stage {stage_id}] Ready — waiting for hidden states...")
        _dash_update(state="waiting", current_step=0)

        times = []
        step = 0
        stage_kv = None
        while True:
            msg = recv_msg(sockets["pull"], timeout_ms=60_000)
            if msg is None or msg["msg_type"] == "end_of_generation":
                break
            if msg["msg_type"] == "trim_cache":
                trim_dynamic_cache(stage_kv, msg["keep_len"])
                continue

            if msg.get("is_prefill", True):
                stage_kv = None

            hidden = tensor_from_activation_msg(msg)
            draft_tokens_list = msg.get("draft_tokens")  # list of K ints or None
            _dash_update(state="forward", current_step=step)
            _publish(sockets["pub"], stage_id, host, step, "forward", 0, step)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits, stage_kv = stage_model(hidden, past_key_values=stage_kv)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            is_spec_step = spec_enabled and not msg.get("is_prefill", True) and draft_tokens_list is not None

            if is_spec_step:
                # Return p(draft_i) for each position + one sample per position
                K = len(draft_tokens_list)
                target_probs, target_samples = [], []
                for i in range(K):
                    p_i = torch.softmax(logits[0, i, :] / max(temperature, 1e-6), dim=-1)
                    target_probs.append(p_i[draft_tokens_list[i]].item())
                    target_samples.append(torch.multinomial(p_i, num_samples=1).item())
                # Bonus: sample from position K (one past last draft)
                p_bonus = torch.softmax(logits[0, K, :] / max(temperature, 1e-6), dim=-1)
                target_samples.append(torch.multinomial(p_bonus, num_samples=1).item())
                send_msg(sockets["token_push"], make_spec_result_msg(step, target_probs, target_samples))
            else:
                # Prefill or non-spec: sample from last position, return as spec_result or plain token
                last_logits = logits[:, -1, :] / max(temperature, 1e-6)
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                if spec_enabled:
                    send_msg(sockets["token_push"],
                             make_spec_result_msg(step, [], [next_token.item()]))
                else:
                    send_msg(sockets["token_push"],
                             make_activation_msg(step, stage_id, next_token.float()))

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
        stage_kv = None
        while True:
            msg = recv_msg(sockets["pull"], timeout_ms=60_000)
            if msg is None or msg["msg_type"] == "end_of_generation":
                send_msg(sockets["push"], make_control_msg("end_of_generation"))
                break
            if msg["msg_type"] == "trim_cache":
                trim_dynamic_cache(stage_kv, msg["keep_len"])
                send_msg(sockets["push"], make_control_msg("trim_cache", keep_len=msg["keep_len"]))
                continue

            is_prefill = msg.get("is_prefill", True)
            draft_tokens_list = msg.get("draft_tokens")
            if is_prefill:
                stage_kv = None

            hidden = tensor_from_activation_msg(msg)
            _dash_update(state="forward", current_step=step)
            _publish(sockets["pub"], stage_id, host, step, "forward", 0, step)

            t0 = time.perf_counter()
            with torch.no_grad():
                out, stage_kv = stage_model(hidden, past_key_values=stage_kv)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            send_msg(sockets["push"], make_activation_msg(
                step, stage_id, out, is_prefill=is_prefill, codec=codec,
                draft_tokens=draft_tokens_list))

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
