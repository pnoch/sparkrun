"""Native (no-container) executor for running workloads directly on hosts.

``NativeExecutor`` runs serve commands as background processes on each
host via nohup, managing them through PID files.  No Docker image,
no volumes, no GPU passthrough — the host environment is used directly.
"""

from __future__ import annotations

import logging

from sparkrun.orchestration.executor import Executor, ExecutorConfig
from sparkrun.utils.shell import b64_wrap_bash, quote

logger = logging.getLogger(__name__)


class NativeExecutor(Executor):
    """Run workloads directly on hosts without containerisation."""

    # Native mode skips the sleep-infinity + exec two-phase launch
    uses_two_phase = False

    def run_cmd(
        self,
        image: str = "",
        command: str = "",
        container_name: str | None = None,
        detach: bool = True,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        extra_opts: list[str] | None = None,
    ) -> str:
        name = container_name or "sparkrun_serve"
        pidfile = "/tmp/%s.pid" % name
        logfile = "/tmp/%s.log" % name

        exports = ""
        if env:
            for key, value in sorted(env.items()):
                exports += "export %s=%s; " % (key, quote(str(value)))

        return (
            "rm -f %(pid)s %(log)s 2>/dev/null || true; "
            "%(exports)snohup bash -c %(cmd)s < /dev/null >> %(log)s 2>&1 & "
            "echo $! > %(pid)s"
        ) % {
            "pid": quote(pidfile),
            "log": quote(logfile),
            "exports": exports,
            "cmd": b64_wrap_bash(command),
        }

    def exec_cmd(
        self,
        container_name: str,
        command: str,
        detach: bool = False,
        env: dict[str, str] | None = None,
    ) -> str:
        name = container_name or "sparkrun_serve"
        logfile = "/tmp/%s.log" % name
        pidfile = "/tmp/%s.pid" % name
        exports = ""
        if env:
            for key, value in sorted(env.items()):
                exports += "export %s=%s; " % (key, quote(str(value)))
        return (
            "rm -f %(pid)s %(log)s 2>/dev/null || true; "
            "%(exports)snohup bash -c %(cmd)s < /dev/null >> %(log)s 2>&1 & "
            "echo $! > %(pid)s"
        ) % {
            "pid": quote(pidfile),
            "log": quote(logfile),
            "exports": exports,
            "cmd": b64_wrap_bash(command),
        }

    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        name = container_name or "sparkrun_serve"
        pidfile = "/tmp/%s.pid" % name
        logfile = "/tmp/%s.log" % name
        return (
            "if [ -f %(pid)s ]; then "
            "pid=$(cat %(pid)s); "
            "kill -15 $pid 2>/dev/null; sleep 1; "
            "kill -9 $pid 2>/dev/null || true; "
            "rm -f %(pid)s; "
            "fi; "
            "rm -f %(log)s 2>/dev/null || true"
        ) % {"pid": quote(pidfile), "log": quote(logfile)}

    def logs_cmd(
        self,
        container_name: str,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        name = container_name or "sparkrun_serve"
        logfile = "/tmp/%s.log" % name
        if follow:
            return "tail -f %s" % quote(logfile)
        if tail is not None:
            return "tail -n %d %s" % (tail, quote(logfile))
        return "cat %s" % quote(logfile)

    def inspect_exists_cmd(self, image: str) -> str:
        return "true"

    def pull_cmd(self, image: str) -> str:
        return "true"

    def generate_direct_serve_script(
        self,
        container_name: str,
        serve_command: str,
        env: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
    ) -> str:
        """Generate a script that directly launches the serve command with nohup.

        For native mode, combines launch + exec into a single nohup call.
        """
        from sparkrun.utils import merge_env

        all_env = merge_env(nccl_env, env)
        return self.run_cmd(
            container_name=container_name,
            command=serve_command,
            env=all_env,
        )
