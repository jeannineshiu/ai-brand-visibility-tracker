"""Prompt runner: queries multiple LLM providers concurrently."""
import asyncio
import uuid
import json
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.table import Table

from src.utils.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, RAW_DIR
from src.utils.models import LLMResponse, PromptConfig
from src.prompt_runner.llm_clients import OpenAIClient, AnthropicClient, GeminiClient, MockClient

console = Console()


def _is_valid_key(key: str, prefix: str) -> bool:
    return bool(key) and key.startswith(prefix) and len(key) > 20 and "..." not in key


def _build_client(provider: str):
    """Return real client if API key looks valid, else MockClient."""
    if provider == "openai":
        return OpenAIClient() if _is_valid_key(OPENAI_API_KEY, "sk-") else MockClient("openai")
    if provider == "anthropic":
        valid = _is_valid_key(ANTHROPIC_API_KEY, "sk-ant-")
        return AnthropicClient() if valid else MockClient("anthropic")
    if provider == "gemini":
        return GeminiClient() if _is_valid_key(GOOGLE_API_KEY, "AIza") else MockClient("gemini")
    raise ValueError("Unknown provider: " + provider)


async def run_prompts(
    prompts: list[PromptConfig],
    providers: Optional[list[str]] = None,
    run_id: Optional[str] = None,
    n_runs: int = 1,
) -> list[LLMResponse]:
    """Query all providers for all prompts concurrently.

    n_runs > 1 runs each (prompt, provider) pair N times for majority voting.
    Each trial gets its own run_id: {base_id}-t0, {base_id}-t1, …
    """
    providers = providers or ["openai", "anthropic", "gemini"]
    base_id = run_id or str(uuid.uuid4())[:8]

    all_responses: list[LLMResponse] = []
    for trial in range(n_runs):
        trial_id = f"{base_id}-t{trial}" if n_runs > 1 else base_id
        clients = {p: _build_client(p) for p in providers}
        tasks = [
            client.query(p.prompt_id, p.prompt_text, trial_id)
            for p in prompts
            for provider, client in clients.items()
        ]
        label = f" (trial {trial + 1}/{n_runs})" if n_runs > 1 else ""
        console.print(f"\n[bold cyan]Run ID:[/] {trial_id}{label}")
        console.print(f"[bold cyan]Prompts:[/] {len(prompts)}  [bold cyan]Providers:[/] {providers}")
        console.print(f"[bold cyan]Total queries:[/] {len(tasks)}\n")

        results: list[LLMResponse] = await asyncio.gather(*tasks)
        _print_summary(results)
        _save_raw(results, trial_id)
        all_responses.extend(results)

    return all_responses


def _print_summary(results: list[LLMResponse]):
    table = Table(title="Query Results")
    table.add_column("Provider", style="cyan")
    table.add_column("Prompt ID")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Status")

    for r in results:
        status = "[green]OK[/]" if not r.error else "[red]ERR[/]"
        table.add_row(
            r.provider.value, r.prompt_id,
            str(r.latency_ms), str(len(r.response_text)), status,
        )

    console.print(table)


def _save_raw(results: list[LLMResponse], run_id: str):
    out_path = RAW_DIR / f"run_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump(mode="json") for r in results], f, indent=2, default=str)
    console.print(f"\n[dim]Saved → {out_path}[/]")
