"""InfiniBand/RDMA detection script generator.

Generates a bash script that detects InfiniBand interfaces and outputs
NCCL/RDMA environment variables. The script is piped to remote hosts
via SSH bash -s.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.scripts import read_script
from sparkrun.utils import parse_kv_output

logger = logging.getLogger(__name__)


@dataclass
class IBDetectionResult:
    """Aggregated IB detection results across multiple hosts.

    Contains NCCL env vars (from head) and per-host IB IP mappings
    for fast internal transfers.
    """

    comm_env: ClusterCommEnv = field(default_factory=ClusterCommEnv.empty)
    """Inter-node comm env derived from IB detection.

    ``comm_env.shared`` holds keys whose values are identical across
    all hosts (``NCCL_NET``, ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``,
    ``UCX_NET_DEVICES``, …).  ``comm_env.per_host`` holds the keys
    that differ between hosts — typically the socket-interface names
    (``GLOO_SOCKET_IFNAME``, ``MN_IF_NAME``, ``TP_SOCKET_IFNAME``,
    ``OMPI_MCA_btl_tcp_if_include``) — so heterogeneous mgmt
    interfaces (e.g. wired on the head, wifi on a worker) don't crash
    gloo at init time.
    """
    ib_ip_map: dict[str, str] = field(default_factory=dict)
    """Mapping of queried host → first IB interface IP.

    Empty for hosts where no IB was detected or no IB IP was found.
    """
    mgmt_ip_map: dict[str, str] = field(default_factory=dict)
    """Mapping of queried host → management interface IP.

    Useful when clusters are defined by IB IPs: lets callers
    display the management IP alongside the IB address.
    """


def generate_ib_detect_script() -> str:
    """Generate a bash script that detects InfiniBand interfaces.

    The script outputs key=value pairs on stdout that can be parsed
    to configure NCCL and RDMA settings for multi-node inference.

    Output variables (if IB is found)::

        DETECTED_GID_INDEX=<n>
        DETECTED_HCA_LIST=<comma-separated HCA names>
        DETECTED_SOCKET_IFNAME=<interface>
        DETECTED_NET_LIST=<comma-separated net interfaces>
        DETECTED_UCX_LIST=<comma-separated UCX devices>
        IB_DETECTED=1

    If no IB is found, outputs::

        IB_DETECTED=0

    Returns:
        Bash script content as a string.
    """
    return read_script("ib_detect.sh")


def parse_ib_detect_output(output: str) -> dict[str, str]:
    """Parse the output of the IB detection script into a dict.

    Args:
        output: Raw stdout from the IB detection script.

    Returns:
        Dictionary of detected key=value pairs.
    """
    return parse_kv_output(output)


def generate_ring_nccl_overrides(ib_info: dict[str, str]) -> dict[str, str]:
    return {
        "NCCL_NET": "Socket",
    }


def generate_nccl_env(ib_info: dict[str, str], topology: str | None = None) -> dict[str, str]:
    if ib_info.get("IB_DETECTED") != "1":
        return {}

    if topology == "ring":
        logger.info("  Applying ring/mesh NCCL overrides (Socket transport)")
        env = {"NCCL_IGNORE_CPU_AFFINITY": "1", "NCCL_NET": "Socket"}
        mgmt_if = ib_info.get("DETECTED_SOCKET_IFNAME", "").strip()
        if mgmt_if:
            env["NCCL_SOCKET_IFNAME"] = mgmt_if.split(",")[0]
        env["NODE_IP"] = ib_info.get("DETECTED_MGMT_IP", "")
        return env

    env: dict[str, str] = {
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_CROSS_NIC": "1",
    }
    if ib_info.get("DETECTED_HCA_LIST"):
        env["NCCL_IB_HCA"] = ib_info["DETECTED_HCA_LIST"]

    def _set_eth_interfaces(target):
        net_list = ib_info[target]
        env["MN_IF_NAME"] = net_list
        env["OMPI_MCA_btl_tcp_if_include"] = net_list
        env["GLOO_SOCKET_IFNAME"] = net_list
        env["TP_SOCKET_IFNAME"] = net_list

    def _nccl_socket_ifname_list(mgmt_if: str | None, ib_nets: str) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        if mgmt_if:
            mgmt_if = mgmt_if.strip()
            if mgmt_if and mgmt_if not in seen:
                parts.append(mgmt_if)
                seen.add(mgmt_if)
        for ifname in (ib_nets or "").split(","):
            ifname = ifname.strip()
            if ifname and ifname not in seen:
                parts.append(ifname)
                seen.add(ifname)
        return ",".join(parts)

    if ib_info.get("DETECTED_SOCKET_IFNAME"):
        _set_eth_interfaces("DETECTED_SOCKET_IFNAME")
        env["NCCL_SOCKET_IFNAME"] = _nccl_socket_ifname_list(
            ib_info["DETECTED_SOCKET_IFNAME"],
            ib_info.get("DETECTED_NET_LIST", ""),
        )
    elif ib_info.get("DETECTED_NET_LIST"):
        _set_eth_interfaces("DETECTED_NET_LIST")
        env["NCCL_SOCKET_IFNAME"] = _nccl_socket_ifname_list(
            None,
            ib_info["DETECTED_NET_LIST"],
        )

    if ib_info.get("DETECTED_UCX_LIST"):
        env["UCX_NET_DEVICES"] = ib_info["DETECTED_UCX_LIST"]

    env["NODE_IP"] = ib_info.get("DETECTED_MGMT_IP", "")

    if ib_info.get("DETECTED_GID_INDEX"):
        env["NCCL_IB_GID_INDEX"] = ib_info["DETECTED_GID_INDEX"]

    return env


def extract_ib_ips(ib_info: dict[str, str]) -> list[str]:
    """Extract InfiniBand interface IPv4 addresses from detection results.

    Args:
        ib_info: Parsed output from :func:`parse_ib_detect_output`.

    Returns:
        List of IB interface IPs (may be empty if no IB or no IPs found).
    """
    raw = ib_info.get("DETECTED_IB_IPS", "")
    if not raw:
        return []
    return [ip.strip() for ip in raw.split(",") if ip.strip()]


def validate_ib_connectivity(
    ib_ip_map: dict[str, str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Validate that the control machine can reach detected IB IPs.

    Tests SSH connectivity from the control machine to each IB IP.
    In ring topologies a host may have IB interfaces on links not
    reachable from the control node, so each IP is tested and the
    first reachable one per host is selected.

    Args:
        ib_ip_map: Mapping of management host → IB IP(s).  Values may
            contain comma-separated IPs (e.g. from ring topologies).
        ssh_kwargs: SSH connection parameters (user, key, options).
        dry_run: Skip the check and return the map unchanged.

    Returns:
        A dict with one reachable IB IP per host.  Returns empty dict
        if none are reachable (signals fallback to management network).
    """
    if not ib_ip_map or dry_run:
        return ib_ip_map

    from sparkrun.orchestration.ssh import run_remote_command

    kw = ssh_kwargs or {}
    reachable: dict[str, str] = {}

    for host, raw_ips in ib_ip_map.items():
        candidates = [ip.strip() for ip in raw_ips.split(",") if ip.strip()]
        picked = None
        for ip in candidates:
            logger.info("Verifying IB reachability for %s (%s)...", host, ip)
            result = run_remote_command(
                ip,
                "true",
                connect_timeout=5,
                timeout=10,
                **kw,
            )
            if result.success:
                picked = ip
                logger.info("  %s: IB reachable at %s", host, ip)
                break
            else:
                logger.info("  %s: IB unreachable at %s, trying next", host, ip)
        if picked:
            reachable[host] = picked
        else:
            logger.warning("  %s: no reachable IB IPs among %s", host, ", ".join(candidates))

    if reachable:
        logger.info("  IB network reachable (%d/%d hosts) — will use IB IPs for transfers", len(reachable), len(ib_ip_map))
        return reachable

    logger.warning(
        "  Control machine cannot reach IB network — falling back to management network for transfers",
    )
    return {}


def detect_ib_for_hosts(
    hosts: list[str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    topology: str | None = None,
) -> IBDetectionResult:
    """Run IB detection on all hosts and return aggregated results.

    Detects InfiniBand on all hosts in parallel, computes NCCL env
    from the head (``hosts[0]``), and builds a mapping of management
    host → first IB IP for use as transfer targets.

    Args:
        hosts: Management hostnames/IPs.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.

    Returns:
        :class:`IBDetectionResult` with NCCL env and IB IP mapping.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not hosts:
        return IBDetectionResult()

    kw = ssh_kwargs or {}
    head_host = hosts[0]

    logger.info("Detecting InfiniBand on %d host(s)...", len(hosts))
    ib_script = generate_ib_detect_script()
    ib_results = run_remote_scripts_parallel(
        hosts,
        ib_script,
        timeout=30,
        dry_run=dry_run,
        **kw,
    )

    per_host_env: dict[str, dict[str, str]] = {}
    ib_ip_map: dict[str, str] = {}
    mgmt_ip_map: dict[str, str] = {}

    for result in ib_results:
        if not result.success:
            continue
        ib_info = parse_ib_detect_output(result.stdout)

        # Per-host comm env (so heterogeneous socket interfaces work)
        host_env = generate_nccl_env(ib_info, topology=topology)
        if host_env:
            per_host_env[result.host] = host_env
            if result.host == head_host:
                logger.info("  InfiniBand detected on %s, comm env configured", head_host)

        # IB IP for transfer routing — store ALL IPs so validate_ib_connectivity
        # can pick the one reachable from the control node (critical for ring topologies)
        ib_ips = extract_ib_ips(ib_info)
        if ib_ips:
            ib_ip_map[result.host] = ",".join(ib_ips)
            logger.debug("  %s IB transfer IPs: %s", result.host, ", ".join(ib_ips))

        # Management IP (from default route interface)
        mgmt_ip = ib_info.get("DETECTED_MGMT_IP", "").strip()
        if mgmt_ip:
            mgmt_ip_map[result.host] = mgmt_ip
            logger.debug("  %s mgmt IP: %s", result.host, mgmt_ip)

    comm_env = ClusterCommEnv.from_per_host(per_host_env)
    if comm_env.is_empty():
        logger.info("  No InfiniBand detected, using default networking")

    if ib_ip_map:
        logger.info("  IB transfer IPs resolved for %d/%d host(s)", len(ib_ip_map), len(hosts))
    else:
        logger.info("  No IB IPs found, transfers will use management network")

    return IBDetectionResult(
        comm_env=comm_env,
        ib_ip_map=ib_ip_map,
        mgmt_ip_map=mgmt_ip_map,
    )
