"""
websocket_bridge.py  —  WebSocket server for web dashboards and HMI.

Serves a WebSocket endpoint that:
  - Broadcasts robot state (odom, joints, detections, RL status) to all clients
  - Accepts commands from web browsers (cmd_vel, arm, goal, e-stop)
  - Serves a built-in HTML dashboard at http://host:port/

Use cases
---------
  Web HMI, tablet remote control, monitoring dashboard,
  simulator browser interface, remote operations centre.

Install
-------
  pip install websockets aiohttp
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Set

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    _WS_OK = True
except ImportError:
    _WS_OK = False

try:
    from aiohttp import web as _aiohttp_web
    _AIOHTTP_OK = True
except ImportError:
    _AIOHTTP_OK = False


# ── built-in dashboard HTML ───────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>P_RoboAI Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #ddd; font-family: monospace; }
  header { background: #1a1a2e; padding: 12px 20px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { color: #7af; font-size: 18px; }
  #status-dot { width: 10px; height: 10px; border-radius: 50%;
                background: #f44; transition: background 0.3s; }
  #status-dot.connected { background: #4f4; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr));
          gap: 12px; padding: 12px; }
  .card { background: #1e1e1e; border: 1px solid #333; border-radius: 8px;
          padding: 14px; }
  .card h2 { color: #7af; font-size: 13px; margin-bottom: 10px;
             text-transform: uppercase; letter-spacing: 1px; }
  .val { color: #4f4; font-size: 22px; font-weight: bold; }
  .sub { color: #888; font-size: 11px; margin-top: 3px; }
  pre { color: #af7; font-size: 11px; overflow: auto; max-height: 140px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  button { background: #2a4a6a; color: #fff; border: none; border-radius: 4px;
           padding: 8px 14px; cursor: pointer; font-family: monospace; }
  button:hover { background: #3a6a9a; }
  button.danger { background: #6a2a2a; }
  button.danger:hover { background: #9a3a3a; }
  input[type=range] { width: 100%; accent-color: #7af; }
  label { font-size: 11px; color: #888; }
  #log { height: 80px; overflow: auto; font-size: 10px; color: #777; }
</style>
</head>
<body>
<header>
  <div id="status-dot"></div>
  <h1>P_RoboAI Live Dashboard</h1>
  <span id="conn-label" style="color:#888;font-size:12px;">Connecting…</span>
</header>

<div class="grid">
  <!-- Odometry -->
  <div class="card">
    <h2>Odometry</h2>
    <div class="val" id="odom-xy">— , —</div>
    <div class="sub">X (m) , Y (m)</div>
    <div style="margin-top:8px">
      <span class="sub">Heading: </span><span id="odom-h" style="color:#af7">—</span>
      <span class="sub"> rad &nbsp; Vx: </span><span id="odom-v" style="color:#af7">—</span>
      <span class="sub"> m/s</span>
    </div>
  </div>

  <!-- Joint States -->
  <div class="card">
    <h2>Joint States</h2>
    <pre id="joints-pre">waiting…</pre>
  </div>

  <!-- YOLO Detections -->
  <div class="card">
    <h2>Detections</h2>
    <div class="val" id="det-count">0</div>
    <div class="sub">objects detected</div>
    <pre id="det-pre" style="margin-top:6px"></pre>
  </div>

  <!-- RL Status -->
  <div class="card">
    <h2>RL Status</h2>
    <pre id="rl-pre">waiting…</pre>
  </div>

  <!-- Robot Control -->
  <div class="card">
    <h2>Control</h2>
    <label>Linear velocity: <span id="lin-val">0.0</span> m/s</label>
    <input type="range" id="lin-slider" min="-2" max="2" step="0.1" value="0"
           oninput="document.getElementById('lin-val').textContent=this.value">
    <label style="margin-top:8px;display:block">
      Angular velocity: <span id="ang-val">0.0</span> rad/s
    </label>
    <input type="range" id="ang-slider" min="-2" max="2" step="0.1" value="0"
           oninput="document.getElementById('ang-val').textContent=this.value">
    <div class="controls">
      <button onclick="sendCmdVel()">Send Cmd</button>
      <button onclick="sendStop()">Stop</button>
      <button class="danger" onclick="sendEstop()">E-STOP</button>
    </div>
  </div>

  <!-- LLM Query -->
  <div class="card">
    <h2>Gemini Query</h2>
    <input id="llm-input" type="text" placeholder="Ask the robot…"
           style="width:100%;background:#2a2a2a;border:1px solid #444;
                  color:#ddd;padding:6px;border-radius:4px;font-family:monospace">
    <div class="controls">
      <button onclick="sendLLMQuery()">Send</button>
    </div>
    <pre id="llm-resp" style="margin-top:6px;color:#fa7"></pre>
  </div>
</div>

<!-- Log -->
<div style="padding:0 12px 12px">
  <div class="card">
    <h2>Event Log</h2>
    <div id="log"></div>
  </div>
</div>

<script>
const WS_URL = `ws://${location.host}/ws`;
let ws = null;

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    document.getElementById('status-dot').classList.add('connected');
    document.getElementById('conn-label').textContent = 'Connected';
    log('Connected to ' + WS_URL);
  };
  ws.onclose = () => {
    document.getElementById('status-dot').classList.remove('connected');
    document.getElementById('conn-label').textContent = 'Reconnecting…';
    log('Disconnected — retry in 3s');
    setTimeout(connect, 3000);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMsg(msg);
  };
}

function handleMsg(msg) {
  if (msg.type === 'odom') {
    document.getElementById('odom-xy').textContent =
      `${msg.x.toFixed(2)}, ${msg.y.toFixed(2)}`;
    document.getElementById('odom-h').textContent = msg.heading.toFixed(3);
    document.getElementById('odom-v').textContent = msg.vx.toFixed(2);
  } else if (msg.type === 'joint_states') {
    const pairs = msg.names.map((n,i) =>
      `${n}: ${msg.positions[i].toFixed(3)}`).join('\\n');
    document.getElementById('joints-pre').textContent = pairs;
  } else if (msg.type === 'detections') {
    document.getElementById('det-count').textContent = msg.count;
    const lines = (msg.detections || []).map(d =>
      `${d.label} (${(d.confidence*100).toFixed(0)}%)`).join('\\n');
    document.getElementById('det-pre').textContent = lines;
  } else if (msg.type === 'rl_status') {
    document.getElementById('rl-pre').textContent = JSON.stringify(msg, null, 2);
  } else if (msg.type === 'llm_response') {
    document.getElementById('llm-resp').textContent = msg.response;
  }
}

function send(obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

function sendCmdVel() {
  send({ type: 'cmd_vel',
    linear:  parseFloat(document.getElementById('lin-slider').value),
    angular: parseFloat(document.getElementById('ang-slider').value) });
}

function sendStop() {
  document.getElementById('lin-slider').value = 0;
  document.getElementById('ang-slider').value = 0;
  document.getElementById('lin-val').textContent = '0.0';
  document.getElementById('ang-val').textContent = '0.0';
  send({ type: 'cmd_vel', linear: 0, angular: 0 });
}

function sendEstop() {
  send({ type: 'estop' });
  log('E-STOP sent!');
}

function sendLLMQuery() {
  const q = document.getElementById('llm-input').value.trim();
  if (q) { send({ type: 'llm_query', query: q }); }
}

function log(msg) {
  const el = document.getElementById('log');
  const t  = new Date().toLocaleTimeString();
  el.innerHTML += `<div>[${t}] ${msg}</div>`;
  el.scrollTop = el.scrollHeight;
}

connect();
</script>
</body>
</html>
"""


# ── WebSocket bridge ──────────────────────────────────────────────────────────

class WebSocketBridge:
    """
    WebSocket server + HTTP dashboard server.

    Parameters
    ----------
    host        : bind address
    ws_port     : WebSocket port (default 8765)
    http_port   : HTTP dashboard port (default 8766)
    on_cmd_vel  : callback(linear, angular)
    on_estop    : callback()
    on_llm      : callback(query)
    """

    def __init__(self,
                 host:       str = "0.0.0.0",
                 ws_port:    int = 8765,
                 http_port:  int = 8766,
                 on_cmd_vel: Optional[Callable] = None,
                 on_estop:   Optional[Callable] = None,
                 on_llm:     Optional[Callable] = None) -> None:
        self._host      = host
        self._ws_port   = ws_port
        self._http_port = http_port
        self._on_cmd    = on_cmd_vel
        self._on_estop  = on_estop
        self._on_llm    = on_llm
        self._clients:  Set = set()
        self._running   = False
        self._loop:     Optional[asyncio.AbstractEventLoop] = None
        self._thread:   Optional[threading.Thread] = None
        self._stats     = {"connected": 0, "messages_sent": 0, "messages_recv": 0}

    @property
    def available(self) -> bool:
        return _WS_OK

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not _WS_OK:
            return False
        self._running = True
        self._thread  = threading.Thread(
            target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        tasks = [self._ws_server()]
        if _AIOHTTP_OK:
            tasks.append(self._http_server())
        await asyncio.gather(*tasks)

    # ── WebSocket server ──────────────────────────────────────────────────────

    async def _ws_server(self) -> None:
        async with websockets.serve(
            self._handle_client, self._host, self._ws_port,
            ping_interval=20, ping_timeout=10,
        ):
            while self._running:
                await asyncio.sleep(0.5)

    async def _handle_client(self,
                               ws: "WebSocketServerProtocol",
                               path: str = "/") -> None:
        self._clients.add(ws)
        self._stats["connected"] = len(self._clients)
        try:
            async for raw in ws:
                self._stats["messages_recv"] += 1
                try:
                    msg = json.loads(raw)
                    await self._handle_command(msg)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            self._stats["connected"] = len(self._clients)

    async def _handle_command(self, msg: dict) -> None:
        t = msg.get("type", "")
        if t == "cmd_vel" and self._on_cmd:
            self._on_cmd(
                float(msg.get("linear",  0.0)),
                float(msg.get("angular", 0.0)),
            )
        elif t == "estop" and self._on_estop:
            self._on_estop()
        elif t == "llm_query" and self._on_llm:
            query = msg.get("query", "")
            if query:
                # Non-blocking — run in executor
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, self._on_llm, query)

    # ── broadcast helpers ─────────────────────────────────────────────────────

    def _broadcast_sync(self, payload: dict) -> None:
        """Thread-safe broadcast from non-async context."""
        if not self._loop or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        msg = json.dumps(payload)
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
                self._stats["messages_sent"] += 1
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def broadcast_odom(self, x: float, y: float,
                        heading: float, vx: float, wz: float) -> None:
        self._broadcast_sync({
            "type": "odom", "x": round(x,3), "y": round(y,3),
            "heading": round(heading,4), "vx": round(vx,3), "wz": round(wz,3),
        })

    def broadcast_joint_states(self, names: list[str],
                                positions: list[float],
                                velocities: list[float]) -> None:
        self._broadcast_sync({
            "type": "joint_states",
            "names":      names,
            "positions":  [round(p,4) for p in positions],
            "velocities": [round(v,4) for v in velocities],
        })

    def broadcast_detections(self, detections: list[dict]) -> None:
        self._broadcast_sync({
            "type": "detections",
            "count": len(detections),
            "detections": detections,
        })

    def broadcast_rl_status(self, episode: int, reward: float,
                             step: int, mode: str) -> None:
        self._broadcast_sync({
            "type": "rl_status",
            "episode": episode, "reward": round(reward,3),
            "step": step, "mode": mode,
        })

    def broadcast_llm_response(self, response: str) -> None:
        self._broadcast_sync({
            "type": "llm_response",
            "response": response[:2000],
        })

    # ── HTTP dashboard ────────────────────────────────────────────────────────

    async def _http_server(self) -> None:
        app = _aiohttp_web.Application()
        app.router.add_get("/", self._http_index)
        app.router.add_get("/health", self._http_health)
        runner = _aiohttp_web.AppRunner(app)
        await runner.setup()
        site = _aiohttp_web.TCPSite(runner, self._host, self._http_port)
        await site.start()
        # Patch the dashboard HTML with the correct WS port
        self._dashboard_html = _DASHBOARD_HTML.replace(
            "location.host", f"'{self._host}:{self._http_port}'",
        )

    async def _http_index(self, request) -> "_aiohttp_web.Response":
        html = _DASHBOARD_HTML
        return _aiohttp_web.Response(
            text=html, content_type="text/html")

    async def _http_health(self, request) -> "_aiohttp_web.Response":
        return _aiohttp_web.Response(
            text=json.dumps(self.get_status()),
            content_type="application/json",
        )

    def get_status(self) -> dict:
        return {
            "available":   _WS_OK,
            "running":     self._running,
            "ws_port":     self._ws_port,
            "http_port":   self._http_port,
            "clients":     len(self._clients),
            "stats":       dict(self._stats),
            "dashboard":   f"http://{self._host}:{self._http_port}/",
        }
