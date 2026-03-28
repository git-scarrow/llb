#!/usr/bin/env python3
import argparse
import asyncio
import json
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Set
import os
from pathlib import Path

import httpx
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


DEFAULT_LB = os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01

# Rendering options (set via CLI flags in main())
SHOW_JSON = False
SHOW_REASONING = False
COMPACT = False


@dataclass
class NodeState:
    name: str
    inflight: int = 0
    maxconn: int = 0
    failures: int = 0
    status: str = "idle"
    lines: List[str] = field(default_factory=list)
    error: Optional[str] = None
    bytes_total: int = 0
    last_chunk_ts: float = 0.0
    rate_bps: float = 0.0
    log_path: Optional[Path] = None
    log_fp: Optional[any] = None

    def _append_text(self, text: str, max_lines: int, compact: bool):
        # Append text to lines, coalescing into the last line in compact mode
        if compact:
            if not self.lines:
                self.lines.append("")
            parts = text.split("\n")
            # append to current last line
            self.lines[-1] += parts[0]
            # for any additional lines, start new entries
            for p in parts[1:]:
                self.lines.append(p)
        else:
            for part in text.splitlines():
                self.lines.append(part)
        if len(self.lines) > max_lines:
            del self.lines[:-max_lines]

    def push(self, s: str, max_lines: int = 100):
        global SHOW_JSON, SHOW_REASONING, COMPACT
        for part in s.splitlines():
            if part.startswith("event: "):
                # Skip SSE event lines from upstream metadata/failover
                continue
            if part.startswith("data: "):
                part = part[6:]
            if not part.strip():
                continue
            if not SHOW_JSON:
                # Try to parse OpenAI-style SSE JSON chunk and extract text
                try:
                    obj = json.loads(part)
                    ch = obj.get("choices")
                    if isinstance(ch, list) and ch:
                        delta = ch[0].get("delta") or {}
                        # Optionally include reasoning content
                        rc = delta.get("reasoning_content")
                        if SHOW_REASONING and rc:
                            self._append_text(rc, max_lines, COMPACT)
                        content = delta.get("content")
                        if content:
                            self._append_text(content, max_lines, COMPACT)
                        # On finish, nothing extra to add (UI shows done when stream ends)
                        continue
                    err = obj.get("error")
                    if err:
                        msg = err.get("message") if isinstance(err, dict) else str(err)
                        self._append_text(f"[error] {msg}", max_lines, COMPACT)
                        continue
                except Exception:
                    # Not JSON; fall through to raw append
                    pass
            # Default: append raw line
            self._append_text(part, max_lines, COMPACT)


async def fetch_json(client: httpx.AsyncClient, url: str):
    r = await client.get(url)
    r.raise_for_status()
    return r.json()


async def get_nodes(client: httpx.AsyncClient, lb: str) -> Dict[str, NodeState]:
    data = await fetch_json(client, f"{lb}/v1/nodes")
    nodes = {}
    for item in data.get("data", []):
        n = item.get("node")
        nodes[n] = NodeState(
            name=n,
            inflight=int(item.get("inflight", 0)),
            maxconn=int(item.get("maxconn", 0)),
            failures=int(item.get("failures", 0)),
        )
    return nodes


async def eligible_nodes_for_model(client: httpx.AsyncClient, lb: str, model: str) -> List[str]:
    qp = httpx.QueryParams({"model": model})
    data = await fetch_json(client, f"{lb}/v1/eligible_nodes?{str(qp)}")
    return data.get("data", [])


async def list_models(client: httpx.AsyncClient, lb: str) -> List[str]:
    data = await fetch_json(client, f"{lb}/v1/models")
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


async def common_models_across_nodes(client: httpx.AsyncClient, lb: str, target_nodes: Set[str], limit: Optional[int] = None) -> List[str]:
    models = await list_models(client, lb)
    if limit:
        models = models[:limit]
    out = []
    for mid in models:
        eligible = set(await eligible_nodes_for_model(client, lb, mid))
        if target_nodes.issubset(eligible):
            out.append(mid)
    return out


def human_rate(bps: float) -> str:
    if bps >= 1024*1024:
        return f"{bps/1024/1024:.1f} MB/s"
    if bps >= 1024:
        return f"{bps/1024:.1f} kB/s"
    return f"{bps:.0f} B/s"


def render_layout(model: str, nodes: Dict[str, NodeState]) -> Layout:
    layout = Layout()
    top = Layout(name="top", size=3)
    grid = Layout(name="grid")
    layout.split_column(top, grid)

    # header table
    t = Table(title=f"LLB Multi-Node Streaming — Model: {model}", show_header=True, header_style="bold")
    t.add_column("Node")
    t.add_column("Inflight")
    t.add_column("MaxConn")
    t.add_column("Failures")
    t.add_column("Status")
    for n in nodes.values():
        t.add_row(n.name, str(n.inflight), str(n.maxconn), str(n.failures), n.status)
    top.update(t)

    # node panels
    grid_children = []
    for n in nodes.values():
        body = Text("\n".join(n.lines[-12:]) or "<no output yet>")
        # colorize rate
        rate = n.rate_bps
        if rate > 4096:
            rate_tag = f"[green]{human_rate(rate)}[/green]"
        elif rate > 1024:
            rate_tag = f"[yellow]{human_rate(rate)}[/yellow]"
        else:
            rate_tag = f"[red]{human_rate(rate)}[/red]"
        title = f"{n.name}  [inflight {n.inflight}/{n.maxconn or '∞'}]  {rate_tag}"
        if n.error:
            title += f"  [error: {n.error}]"
        grid_children.append(Panel(body, title=title))
    # arrange as rows of up to 2 per row for simplicity
    grid.split_row(*grid_children) if grid_children else None
    return layout


async def stream_node(client: httpx.AsyncClient, lb: str, node: NodeState, model: str, prompt: str):
    url = f"{lb}/v1/chat/completions?node={node.name}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    node.status = "connecting"
    try:
        async with client.stream("POST", url, json=body) as resp:
            node.status = f"{resp.status_code}"
            async for chunk in resp.aiter_text():
                node.push(chunk)
                now = time.monotonic()
                sz = len(chunk.encode(errors="ignore"))
                if node.last_chunk_ts > 0:
                    dt = max(1e-3, now - node.last_chunk_ts)
                    inst = sz / dt
                    node.rate_bps = 0.7 * node.rate_bps + 0.3 * inst
                else:
                    node.rate_bps = node.rate_bps or float(sz)
                node.last_chunk_ts = now
                node.bytes_total += sz
                if node.log_fp:
                    try:
                        node.log_fp.write(chunk)
                        if not chunk.endswith("\n"):
                            node.log_fp.write("\n")
                        node.log_fp.flush()
                    except Exception:
                        pass
    except Exception as e:
        node.error = str(e)
        node.status = "error"


async def refresh_stats(client: httpx.AsyncClient, lb: str, nodes: Dict[str, NodeState]):
    try:
        data = await fetch_json(client, f"{lb}/v1/nodes")
        index = {item.get("node"): item for item in data.get("data", [])}
        for name, st in nodes.items():
            info = index.get(name)
            if info:
                st.inflight = int(info.get("inflight", st.inflight))
                st.failures = int(info.get("failures", st.failures))
                st.maxconn = int(info.get("maxconn", st.maxconn))
    except Exception:
        pass


def _load_state(path: Path) -> Dict:
    try:
        if path.is_file():
            return json.loads(path.read_text())
    except Exception:
        return {}
    return {}


def _save_state(path: Path, lb: str, model: str, nodes: List[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"lb": lb, "model": model, "nodes": nodes}
        path.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _safe_name(node: str) -> str:
    return node.replace(":", "_").replace("/", "_")


async def run(lb: str, model: Optional[str], prompt: str, include: List[str], exclude: List[str], pick_nodes: bool, pick_model: bool, require_all_nodes: bool, filter_models: Optional[str], state_file: Optional[Path], no_restore: bool, save_logs_dir: Optional[Path], linger_secs: float = 0.0, wait_on_exit: bool = False):
    console = Console()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        healthy = await get_nodes(client, lb)
        # Filter by include/exclude
        names = list(healthy.keys())
        if include:
            include_set = set(include)
            names = [n for n in names if n in include_set]
        if exclude:
            exclude_set = set(exclude)
            names = [n for n in names if n not in exclude_set]
        if not names:
            console.print("[red]No nodes selected after include/exclude filtering[/red]")
            return

        # Restore from state if requested and no explicit include/exclude/picker was used
        if state_file and not no_restore and not include and not exclude and not pick_nodes:
            prev = _load_state(state_file)
            prev_nodes = prev.get("nodes") if isinstance(prev, dict) else None
            if prev_nodes:
                restored = [n for n in prev_nodes if n in names]
                if restored:
                    console.print(f"[cyan]Restored nodes from state:[/cyan] {', '.join(restored)}")
                    names = restored

        # Optional node picker
        if pick_nodes:
            console.print("Select nodes by index (comma-separated). Available:")
            for i, n in enumerate(names):
                console.print(f"  [{i}] {n}")
            try:
                selection = input("Nodes (e.g., 0,2): ").strip()
                try:
                    idx = {int(x) for x in selection.split(',') if x.strip()}
                    names = [names[i] for i in sorted(idx) if 0 <= i < len(names)]
                except Exception:
                    console.print("[yellow]Invalid selection; keeping all[/yellow]")
            except EOFError:
                console.print("[yellow]No TTY; keeping all nodes[/yellow]")

        target_nodes = set(names)

        # Model resolution / picker
        if not model or pick_model:
            # restore last model if possible
            if state_file and not no_restore and not pick_model and not model:
                prev = _load_state(state_file)
                if isinstance(prev, dict) and prev.get("model"):
                    model = prev.get("model")
            if require_all_nodes:
                console.print("Computing models common to all selected nodes...")
                commons = await common_models_across_nodes(client, lb, target_nodes)
                if filter_models:
                    f = filter_models.lower()
                    commons = [m for m in commons if f in m.lower()]
                if not commons:
                    console.print("[red]No common models across selected nodes[/red]")
                    return
                for i, mid in enumerate(commons):
                    console.print(f"  [{i}] {mid}")
                try:
                    sel = input("Model index: ").strip()
                    try:
                        model = commons[int(sel)]
                    except Exception:
                        model = commons[0]
                except EOFError:
                    model = commons[0]
            else:
                # Just let user pick from global list
                all_models = await list_models(client, lb)
                if filter_models:
                    f = filter_models.lower()
                    all_models = [m for m in all_models if f in m.lower()]
                for i, mid in enumerate(all_models[:200]):
                    console.print(f"  [{i}] {mid}")
                try:
                    sel = input("Model index: ").strip()
                    try:
                        model = all_models[int(sel)]
                    except Exception:
                        model = all_models[0] if all_models else None
                except EOFError:
                    model = all_models[0] if all_models else None
        if not model:
            console.print("[red]No model selected[/red]")
            return

        # Only stream to nodes that both are healthy and have the model
        eligible = set(await eligible_nodes_for_model(client, lb, model))
        names = [n for n in names if n in eligible]
        nodes = {name: healthy[name] for name in names}
        if not nodes:
            console.print(f"[red]No eligible nodes for model {model}[/red]")
            return

        # Setup logging if requested
        if save_logs_dir:
            try:
                save_logs_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                for name, st in nodes.items():
                    st.log_path = save_logs_dir / f"{_safe_name(name)}.{ts}.log"
                    try:
                        st.log_fp = st.log_path.open("a", encoding="utf-8")
                    except Exception:
                        st.log_fp = None
            except Exception:
                pass

        # Launch stream tasks
        tasks = [asyncio.create_task(stream_node(client, lb, node, model, prompt)) for node in nodes.values()]

        with Live(render_layout(model, nodes), console=console, refresh_per_second=6):
            try:
                while any(not t.done() for t in tasks):
                    await refresh_stats(client, lb, nodes)
                    await asyncio.sleep(0.15)
            except KeyboardInterrupt:
                for t in tasks:
                    t.cancel()

        # Await completion quietly
        for t in tasks:
            try:
                await t
            except Exception:
                pass

        # Close logs
        for st in nodes.values():
            try:
                if st.log_fp:
                    st.log_fp.close()
            except Exception:
                pass

        # Persist state
        if state_file:
            _save_state(state_file, lb, model, list(nodes.keys()))

        # Show a static snapshot of the final layout/content before exiting
        try:
            console.print(render_layout(model, nodes))
        except Exception:
            try:
                # Fallback: minimal per-node dump
                for n in nodes.values():
                    console.print(Panel(Text("\n".join(n.lines[-20:]) or "<no output>").append("\n"), title=f"{n.name}"))
            except Exception:
                pass

        # Optional linger or wait before exit so users can view the final screen
        try:
            if wait_on_exit:
                console.print("\n[dim]Press Enter to exit...[/dim]")
                try:
                    # Avoid blocking the event loop
                    await asyncio.to_thread(input)
                except EOFError:
                    pass
            elif linger_secs and linger_secs > 0:
                await asyncio.sleep(linger_secs)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Sleek TUI to stream chat through eligible nodes")
    ap.add_argument("--lb", default=DEFAULT_LB, help="Load balancer base URL")
    ap.add_argument("--model", help="Model ID to use (omit with --pick-model)")
    ap.add_argument("--prompt", required=True, help="User prompt/message")
    ap.add_argument("--include", help="Comma-separated nodes to include", default="")
    ap.add_argument("--exclude", help="Comma-separated nodes to exclude", default="")
    ap.add_argument("--pick-model", action="store_true", help="Interactive model picker")
    ap.add_argument("--pick-nodes", action="store_true", help="Interactive node picker")
    ap.add_argument("--require-all-nodes", action="store_true", help="Only list models available on all selected nodes")
    ap.add_argument("--filter-models", help="Substring to filter model list", default="")
    ap.add_argument("--state-file", default=str(Path.home()/".llb"/"tui_state.json"), help="Path to persist last selection")
    ap.add_argument("--no-restore", action="store_true", help="Do not restore last selection from state file")
    ap.add_argument("--save-logs", help="Directory to save per-node stream logs")
    ap.add_argument("--linger-secs", type=float, default=0.0, help="Keep the UI visible for N seconds after streams finish")
    ap.add_argument("--wait-on-exit", action="store_true", help="Wait for Enter before exiting (keeps UI visible)")
    ap.add_argument("--show-json", action="store_true", help="Show raw SSE JSON lines instead of extracted text")
    ap.add_argument("--show-reasoning", action="store_true", help="Include reasoning_content chunks when present")
    ap.add_argument("--compact", action="store_true", help="Coalesce tokens into flowing text instead of per-line tokens")
    args = ap.parse_args()

    include = [x.strip() for x in args.include.split(',') if x.strip()]
    exclude = [x.strip() for x in args.exclude.split(',') if x.strip()]

    state_file = Path(args.state_file) if args.state_file else None
    save_logs_dir = Path(args.save_logs) if args.save_logs else None
    # Set rendering globals
    global SHOW_JSON, SHOW_REASONING, COMPACT
    SHOW_JSON = bool(args.show_json)
    SHOW_REASONING = bool(args.show_reasoning)
    COMPACT = bool(args.compact)
    asyncio.run(run(args.lb, args.model, args.prompt, include, exclude, args.pick_nodes, args.pick_model, args.require_all_nodes, args.filter_models or None, state_file, args.no_restore, save_logs_dir, args.linger_secs, args.wait_on_exit))


if __name__ == "__main__":
    main()
