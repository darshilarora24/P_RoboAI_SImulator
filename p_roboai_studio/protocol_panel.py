"""
protocol_panel.py  —  Protocol Bridge panel for P_RoboAI Studio.

Qt dock widget providing a control surface for all protocol bridges:
  DDS | OPC UA | Modbus | CAN/CANopen | EtherCAT | PROFINET | MQTT | gRPC | WebSocket

Each protocol has its own tab with connection controls and live status.
All bridges run in background threads so the GUI stays responsive.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore    import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui     import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QTabWidget, QGroupBox, QComboBox,
    QCheckBox, QSpinBox, QFormLayout, QScrollArea, QFrame,
    QSplitter,
)

# Bootstrap .venv so bridge imports work from studio
for _root in Path(__file__).resolve().parents:
    _venv = _root / ".venv"
    if _venv.exists():
        import site as _site
        for _sp in _venv.glob("lib/python*/site-packages"):
            if str(_sp) not in sys.path:
                _site.addsitedir(str(_sp))
                sys.path.insert(0, str(_sp))
        break

try:
    from dds_config       import DDSConfig
    from opcua_bridge     import OPCUABridge
    from modbus_bridge    import ModbusBridge
    from canopen_bridge   import CANopenBridge
    from ethercat_bridge  import EtherCATBridge
    from profinet_bridge  import PROFINETDiscovery, S7Bridge, generate_gsdml
    from mqtt_bridge      import MQTTBridge
    from grpc_bridge      import GRPCBridge
    from websocket_bridge import WebSocketBridge
    _BRIDGES_OK = True
except ImportError:
    _BRIDGES_OK = False


# ── status indicator ──────────────────────────────────────────────────────────

def _dot(connected: bool) -> str:
    return "🟢" if connected else "🔴"


# ── protocol status widget ────────────────────────────────────────────────────

class _StatusBox(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(120)
        self.setFont(QFont("monospace", 8))
        self.setStyleSheet(
            "QTextEdit{background:#1a1a1a;color:#7af;"
            "border:1px solid #333;border-radius:4px;}")

    def set_status(self, data: dict) -> None:
        self.setPlainText(json.dumps(data, indent=2))


# ── individual protocol tabs ──────────────────────────────────────────────────

class _DDSTab(QWidget):
    def __init__(self, dds: "DDSConfig") -> None:
        super().__init__()
        self._dds = dds
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        grp = QGroupBox("RMW Implementation")
        form = QFormLayout(grp)
        self._rmw_combo = QComboBox()
        self._rmw_combo.addItems(["fastdds", "cyclone", "connext"])
        self._rmw_combo.currentTextChanged.connect(
            lambda t: dds.set_rmw(t))
        form.addRow("DDS backend:", self._rmw_combo)
        lay.addWidget(grp)

        grp2 = QGroupBox("Profile")
        row  = QHBoxLayout(grp2)
        write_fast = QPushButton("Write Fast DDS Profile")
        write_fast.clicked.connect(lambda: dds.write_fastdds_profile())
        write_cyc  = QPushButton("Write Cyclone Config")
        write_cyc.clicked.connect(lambda: dds.write_cyclone_config())
        row.addWidget(write_fast)
        row.addWidget(write_cyc)
        lay.addWidget(grp2)

        self._status = _StatusBox()
        lay.addWidget(QLabel("Current config:"))
        lay.addWidget(self._status)
        lay.addStretch()
        self.refresh()

    def refresh(self) -> None:
        self._status.set_status(self._dds.get_status())


class _OPCUATab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[OPCUABridge] = None
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        grp = QGroupBox("OPC UA Server")
        form = QFormLayout(grp)
        self._ep_edit = QLineEdit("opc.tcp://0.0.0.0:4840/p_roboai")
        form.addRow("Endpoint:", self._ep_edit)
        self._client_edit = QLineEdit()
        self._client_edit.setPlaceholderText("opc.tcp://plc-host:4840 (optional)")
        form.addRow("Client URL:", self._client_edit)
        row = QHBoxLayout()
        self._start_btn = QPushButton("Start Server")
        self._start_btn.clicked.connect(self._start)
        self._stop_btn  = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setEnabled(False)
        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)
        form.addRow(row)
        lay.addWidget(grp)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()

    def _start(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install asyncua")
            return
        self._bridge = OPCUABridge(
            endpoint=self._ep_edit.text().strip(),
            client_url=self._client_edit.text().strip(),
        )
        threading.Thread(
            target=lambda: __import__("asyncio").run(
                self._bridge.start_server()), daemon=True).start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _stop(self) -> None:
        if self._bridge:
            threading.Thread(
                target=lambda: __import__("asyncio").run(
                    self._bridge.stop_server()), daemon=True).start()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def refresh(self) -> None:
        if self._bridge:
            self._status.set_status(self._bridge.get_status())


class _ModbusTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[ModbusBridge] = None
        lay = QVBoxLayout(self)

        grp = QGroupBox("Modbus TCP Server")
        form = QFormLayout(grp)
        self._host_edit = QLineEdit("0.0.0.0")
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(5020)
        form.addRow("Host:", self._host_edit)
        form.addRow("Port:", self._port_spin)
        row = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._start)
        self._stop_btn  = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop)
        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)
        form.addRow(row)
        lay.addWidget(grp)

        grp2 = QGroupBox("Modbus Client — Read External Device")
        form2 = QFormLayout(grp2)
        self._cl_host = QLineEdit()
        self._cl_host.setPlaceholderText("192.168.1.50")
        self._cl_port = QSpinBox()
        self._cl_port.setRange(1, 65535)
        self._cl_port.setValue(502)
        self._cl_start = QSpinBox()
        self._cl_count = QSpinBox()
        self._cl_count.setRange(1, 125)
        self._cl_count.setValue(10)
        read_btn = QPushButton("Read")
        read_btn.clicked.connect(self._client_read)
        form2.addRow("Host:", self._cl_host)
        form2.addRow("Port:", self._cl_port)
        form2.addRow("Start reg:", self._cl_start)
        form2.addRow("Count:", self._cl_count)
        form2.addRow(read_btn)
        lay.addWidget(grp2)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()

    def _start(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install pymodbus")
            return
        self._bridge = ModbusBridge(
            host=self._host_edit.text(),
            port=self._port_spin.value(),
        )
        ok = self._bridge.start()
        self._status.setPlainText(
            f"Modbus server {'started' if ok else 'FAILED — check permissions'}")

    def _stop(self) -> None:
        if self._bridge:
            self._bridge.stop()

    def _client_read(self) -> None:
        if not _BRIDGES_OK:
            return
        tmp = ModbusBridge()
        result = tmp.read_holding_registers(
            self._cl_host.text(), self._cl_port.value(),
            self._cl_start.value(), self._cl_count.value())
        self._status.setPlainText(
            f"Registers: {result}" if result else "Read failed")

    def refresh(self) -> None:
        if self._bridge:
            self._status.set_status(self._bridge.get_status())


class _CANTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[CANopenBridge] = None
        lay = QVBoxLayout(self)

        grp = QGroupBox("CAN Bus / CANopen")
        form = QFormLayout(grp)
        self._iface_combo = QComboBox()
        self._iface_combo.addItems(["socketcan", "pcan", "kvaser", "vector", "virtual"])
        self._chan_edit = QLineEdit("can0")
        self._brate_combo = QComboBox()
        self._brate_combo.addItems(["125000", "250000", "500000", "1000000"])
        self._brate_combo.setCurrentText("500000")
        form.addRow("Interface:", self._iface_combo)
        form.addRow("Channel:", self._chan_edit)
        form.addRow("Bitrate:", self._brate_combo)
        row = QHBoxLayout()
        conn_btn = QPushButton("Connect")
        conn_btn.clicked.connect(self._connect)
        disc_btn = QPushButton("Disconnect")
        disc_btn.clicked.connect(self._disconnect)
        row.addWidget(conn_btn)
        row.addWidget(disc_btn)
        form.addRow(row)
        lay.addWidget(grp)

        grp2 = QGroupBox("NMT Control")
        nmt_row = QHBoxLayout(grp2)
        for label, fn in [("Start", "nmt_start"), ("Stop", "nmt_stop"),
                           ("PreOp", "nmt_preop"), ("Reset", "nmt_reset")]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _, f=fn: self._bridge and getattr(self._bridge, f)())
            nmt_row.addWidget(btn)
        lay.addWidget(grp2)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()

    def _connect(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install python-can canopen")
            return
        self._bridge = CANopenBridge(
            interface=self._iface_combo.currentText(),
            channel=self._chan_edit.text(),
            bitrate=int(self._brate_combo.currentText()),
        )
        ok = self._bridge.connect()
        self._status.setPlainText(
            f"CANopen {'connected' if ok else 'FAILED — check interface'}")

    def _disconnect(self) -> None:
        if self._bridge:
            self._bridge.disconnect()

    def refresh(self) -> None:
        if self._bridge:
            self._status.set_status(self._bridge.get_status())


class _EtherCATTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[EtherCATBridge] = None
        lay = QVBoxLayout(self)

        grp = QGroupBox("EtherCAT Master")
        form = QFormLayout(grp)
        self._iface_edit = QLineEdit("eth0")
        self._cycle_spin = QSpinBox()
        self._cycle_spin.setRange(100, 100000)
        self._cycle_spin.setValue(1000)
        self._cycle_spin.setSuffix(" μs")
        form.addRow("Network interface:", self._iface_edit)
        form.addRow("Cycle time:", self._cycle_spin)
        row = QHBoxLayout()
        init_btn  = QPushButton("Init Master")
        init_btn.clicked.connect(self._init)
        start_btn = QPushButton("Start Cyclic")
        start_btn.clicked.connect(self._start)
        stop_btn  = QPushButton("Stop")
        stop_btn.clicked.connect(self._stop)
        row.addWidget(init_btn)
        row.addWidget(start_btn)
        row.addWidget(stop_btn)
        form.addRow(row)
        lay.addWidget(grp)

        note = QLabel(
            "⚠ EtherCAT requires CAP_NET_RAW permission and SOEM-compatible NIC.\n"
            "  pip install pysoem\n"
            "  sudo setcap cap_net_raw+ep $(which python3)")
        note.setStyleSheet("color:#fa7;font-size:10px;")
        note.setWordWrap(True)
        lay.addWidget(note)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()

    def _init(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install pysoem")
            return
        self._bridge = EtherCATBridge(
            interface=self._iface_edit.text(),
            cycle_us=self._cycle_spin.value(),
        )
        ok = self._bridge.init()
        self._status.setPlainText(
            f"EtherCAT init {'OK' if ok else 'FAILED — check NIC and permissions'}")

    def _start(self) -> None:
        if self._bridge:
            self._bridge.start_cyclic()
            self._status.setPlainText("Cyclic PDO exchange running")

    def _stop(self) -> None:
        if self._bridge:
            self._bridge.close()
            self._status.setPlainText("Stopped")

    def refresh(self) -> None:
        if self._bridge:
            self._status.set_status(self._bridge.get_status())


class _PROFINETTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)

        grp = QGroupBox("PROFINET Discovery")
        form = QFormLayout(grp)
        self._iface_edit = QLineEdit("eth0")
        form.addRow("Interface:", self._iface_edit)
        row = QHBoxLayout()
        disc_btn = QPushButton("Start Discovery")
        disc_btn.clicked.connect(self._start_discovery)
        gsdml_btn = QPushButton("Generate GSDML…")
        gsdml_btn.clicked.connect(self._gen_gsdml)
        row.addWidget(disc_btn)
        row.addWidget(gsdml_btn)
        form.addRow(row)
        lay.addWidget(grp)

        grp2 = QGroupBox("Siemens S7 PLC (snap7)")
        form2 = QFormLayout(grp2)
        self._plc_ip   = QLineEdit()
        self._plc_ip.setPlaceholderText("192.168.1.10")
        self._plc_rack = QSpinBox()
        self._plc_slot = QSpinBox()
        self._plc_slot.setValue(1)
        form2.addRow("PLC IP:", self._plc_ip)
        form2.addRow("Rack:", self._plc_rack)
        form2.addRow("Slot:", self._plc_slot)
        conn_btn = QPushButton("Connect to PLC")
        conn_btn.clicked.connect(self._connect_plc)
        form2.addRow(conn_btn)
        lay.addWidget(grp2)

        note = QLabel(
            "Install: pip install python-snap7\n"
            "PLC must have PUT/GET enabled in TIA Portal.")
        note.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(note)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()
        self._disc = None
        self._s7   = None

    def _start_discovery(self) -> None:
        if not _BRIDGES_OK:
            return
        self._disc = PROFINETDiscovery(
            interface=self._iface_edit.text(),
            on_device=lambda d: self._status.setPlainText(
                f"Found: {json.dumps(d, indent=2)}"),
        )
        ok = self._disc.start()
        if ok:
            self._disc.send_identify_request()
            self._status.setPlainText("Discovery started — listening for DCP responses…")
        else:
            self._status.setPlainText("Discovery failed (needs CAP_NET_RAW or root)")

    def _gen_gsdml(self) -> None:
        if not _BRIDGES_OK:
            return
        path = generate_gsdml(output_path="/tmp/P_RoboAI.xml")
        self._status.setPlainText(f"GSDML written: {path}")

    def _connect_plc(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install python-snap7")
            return
        self._s7 = S7Bridge(
            ip=self._plc_ip.text(),
            rack=self._plc_rack.value(),
            slot=self._plc_slot.value(),
        )
        ok = self._s7.connect()
        self._status.setPlainText(
            f"S7 {'connected' if ok else 'FAILED — check IP and PUT/GET setting'}")

    def refresh(self) -> None:
        if self._s7:
            self._status.set_status(self._s7.get_status())


class _MQTTTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[MQTTBridge] = None
        lay = QVBoxLayout(self)

        grp = QGroupBox("MQTT Broker")
        form = QFormLayout(grp)
        self._broker  = QLineEdit("localhost")
        self._port    = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(1883)
        self._robot_id = QLineEdit("robot_01")
        self._user    = QLineEdit()
        self._pass    = QLineEdit()
        self._pass.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Broker:", self._broker)
        form.addRow("Port:", self._port)
        form.addRow("Robot ID:", self._robot_id)
        form.addRow("Username:", self._user)
        form.addRow("Password:", self._pass)
        row = QHBoxLayout()
        conn_btn = QPushButton("Connect")
        conn_btn.clicked.connect(self._connect)
        disc_btn = QPushButton("Disconnect")
        disc_btn.clicked.connect(self._disconnect)
        row.addWidget(conn_btn)
        row.addWidget(disc_btn)
        form.addRow(row)
        lay.addWidget(grp)

        grp2 = QGroupBox("Test Publish")
        pub_row = QHBoxLayout(grp2)
        self._pub_topic = QLineEdit("status")
        self._pub_msg   = QLineEdit('{"test": true}')
        pub_btn = QPushButton("Publish")
        pub_btn.clicked.connect(self._test_publish)
        pub_row.addWidget(QLabel("Suffix:"))
        pub_row.addWidget(self._pub_topic)
        pub_row.addWidget(QLabel("Payload:"))
        pub_row.addWidget(self._pub_msg)
        pub_row.addWidget(pub_btn)
        lay.addWidget(grp2)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()

    def _connect(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install paho-mqtt")
            return
        self._bridge = MQTTBridge(
            broker=self._broker.text(),
            port=self._port.value(),
            robot_id=self._robot_id.text(),
            username=self._user.text(),
            password=self._pass.text(),
        )
        ok = self._bridge.connect()
        self._status.setPlainText(
            f"MQTT {'connected' if ok else 'FAILED — check broker address'}")

    def _disconnect(self) -> None:
        if self._bridge:
            self._bridge.disconnect()

    def _test_publish(self) -> None:
        if not self._bridge:
            return
        try:
            payload = json.loads(self._pub_msg.text())
            self._bridge.publish(self._pub_topic.text(), payload)
        except json.JSONDecodeError:
            self._bridge.publish(self._pub_topic.text(),
                                  {"msg": self._pub_msg.text()})

    def refresh(self) -> None:
        if self._bridge:
            self._status.set_status(self._bridge.get_status())


class _GRPCTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[GRPCBridge] = None
        lay = QVBoxLayout(self)

        grp = QGroupBox("gRPC Server")
        form = QFormLayout(grp)
        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(50051)
        form.addRow("Port:", self._port)
        row = QHBoxLayout()
        start_btn = QPushButton("Start Server")
        start_btn.clicked.connect(self._start)
        stop_btn  = QPushButton("Stop")
        stop_btn.clicked.connect(self._stop)
        row.addWidget(start_btn)
        row.addWidget(stop_btn)
        form.addRow(row)
        lay.addWidget(grp)

        gen_grp = QGroupBox("Generate gRPC Stubs")
        gen_row = QHBoxLayout(gen_grp)
        gen_btn = QPushButton("Run protoc (generate Python stubs)")
        gen_btn.clicked.connect(self._gen_stubs)
        gen_row.addWidget(gen_btn)
        lay.addWidget(gen_grp)

        note = QLabel(
            "Install: pip install grpcio grpcio-tools\n"
            "Proto file: src/p_roboai_protocols/proto/robot_service.proto\n"
            "Services: SendCmdVel, SendArmCommand, QueryLLM, EStop,\n"
            "  StreamOdometry, StreamJointStates, StreamDetections, TeleOpStream")
        note.setStyleSheet("color:#888;font-size:10px;")
        note.setWordWrap(True)
        lay.addWidget(note)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()

    def _start(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install grpcio grpcio-tools")
            return
        self._bridge = GRPCBridge(port=self._port.value())
        ok = self._bridge.start()
        self._status.setPlainText(
            f"gRPC server {'started on port ' + str(self._port.value()) if ok else 'FAILED'}")

    def _stop(self) -> None:
        if self._bridge:
            self._bridge.stop()

    def _gen_stubs(self) -> None:
        import subprocess, os
        proto = str(Path(__file__).parent.parent /
                    "src/p_roboai_protocols/proto/robot_service.proto")
        out   = str(Path(__file__).parent.parent /
                    "src/p_roboai_protocols/p_roboai_protocols")
        cmd   = [sys.executable, "-m", "grpc_tools.protoc",
                 f"-I{Path(proto).parent}",
                 f"--python_out={out}",
                 f"--grpc_python_out={out}",
                 proto]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            self._status.setPlainText("Stubs generated successfully!")
        else:
            self._status.setPlainText(
                f"protoc failed:\n{result.stderr}\n"
                "Ensure grpcio-tools is installed.")

    def refresh(self) -> None:
        if self._bridge:
            self._status.set_status(self._bridge.get_status())


class _WebSocketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[WebSocketBridge] = None
        lay = QVBoxLayout(self)

        grp = QGroupBox("WebSocket + HTTP Dashboard")
        form = QFormLayout(grp)
        self._ws_port   = QSpinBox()
        self._ws_port.setRange(1, 65535)
        self._ws_port.setValue(8765)
        self._http_port = QSpinBox()
        self._http_port.setRange(1, 65535)
        self._http_port.setValue(8766)
        form.addRow("WebSocket port:", self._ws_port)
        form.addRow("HTTP dashboard port:", self._http_port)
        row = QHBoxLayout()
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self._start)
        stop_btn  = QPushButton("Stop")
        stop_btn.clicked.connect(self._stop)
        open_btn  = QPushButton("Open Dashboard ↗")
        open_btn.clicked.connect(self._open_dashboard)
        row.addWidget(start_btn)
        row.addWidget(stop_btn)
        row.addWidget(open_btn)
        form.addRow(row)
        lay.addWidget(grp)

        note = QLabel(
            "Install: pip install websockets aiohttp\n"
            "Dashboard: http://localhost:8766/\n"
            "WebSocket: ws://localhost:8765/ws")
        note.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(note)

        self._status = _StatusBox()
        lay.addWidget(self._status)
        lay.addStretch()

    def _start(self) -> None:
        if not _BRIDGES_OK:
            self._status.setPlainText("Install: pip install websockets aiohttp")
            return
        self._bridge = WebSocketBridge(
            ws_port=self._ws_port.value(),
            http_port=self._http_port.value(),
        )
        ok = self._bridge.start()
        if ok:
            self._status.setPlainText(
                f"WebSocket: ws://localhost:{self._ws_port.value()}\n"
                f"Dashboard: http://localhost:{self._http_port.value()}/")

    def _stop(self) -> None:
        if self._bridge:
            self._bridge.stop()

    def _open_dashboard(self) -> None:
        import webbrowser
        webbrowser.open(f"http://localhost:{self._http_port.value()}/")

    def refresh(self) -> None:
        if self._bridge:
            self._status.set_status(self._bridge.get_status())


# ── main panel ────────────────────────────────────────────────────────────────

class ProtocolPanel(QWidget):
    """Protocol Bridge panel — hosts all 9 protocol tabs."""

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_all)
        self._timer.start(2000)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(4, 4, 4, 4)

        if not _BRIDGES_OK:
            warn = QLabel(
                "Protocol bridges not installed in current Python path.\n"
                "Run: pip install asyncua pymodbus python-can canopen "
                "pysoem python-snap7 paho-mqtt grpcio grpcio-tools "
                "websockets aiohttp")
            warn.setStyleSheet("color:#fa7;font-size:11px;")
            warn.setWordWrap(True)
            root.addWidget(warn)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabBar::tab{padding:4px 10px;font-size:11px;}"
            "QTabBar::tab:selected{background:#1a3a5a;color:#7af;}")

        self._dds_tab  = _DDSTab(DDSConfig() if _BRIDGES_OK else _FallbackDDS())
        self._opcua    = _OPCUATab()
        self._modbus   = _ModbusTab()
        self._can      = _CANTab()
        self._ecat     = _EtherCATTab()
        self._profinet = _PROFINETTab()
        self._mqtt     = _MQTTTab()
        self._grpc     = _GRPCTab()
        self._ws       = _WebSocketTab()

        tabs.addTab(self._dds_tab,  "DDS")
        tabs.addTab(self._opcua,    "OPC UA")
        tabs.addTab(self._modbus,   "Modbus")
        tabs.addTab(self._can,      "CAN")
        tabs.addTab(self._ecat,     "EtherCAT")
        tabs.addTab(self._profinet, "PROFINET")
        tabs.addTab(self._mqtt,     "MQTT")
        tabs.addTab(self._grpc,     "gRPC")
        tabs.addTab(self._ws,       "WebSocket")
        root.addWidget(tabs)

        # Global status bar
        self._global_status = QLabel("All protocols stopped")
        self._global_status.setStyleSheet(
            "color:#888;font-size:10px;padding:2px 4px;")
        root.addWidget(self._global_status)

    def _refresh_all(self) -> None:
        for tab in [self._dds_tab, self._opcua, self._modbus, self._can,
                    self._ecat, self._profinet, self._mqtt, self._grpc, self._ws]:
            try:
                tab.refresh()
            except Exception:
                pass


class _FallbackDDS:
    """Stub when bridges not available."""
    def set_rmw(self, _): pass
    def write_fastdds_profile(self): pass
    def write_cyclone_config(self): pass
    def get_status(self):
        return {"status": "bridges not installed"}
