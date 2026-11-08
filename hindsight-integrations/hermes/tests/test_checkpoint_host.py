"""Optional contract check against an installed Hermes source tree, without an LLM."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("HERMES_AGENT_SOURCE"), reason="set HERMES_AGENT_SOURCE for real host coverage")
def test_real_host_checkpoint_gate_preserves_normalization_redaction_and_failure(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("HINDSIGHT_", "HERMES_"))}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path), HERMES_AGENT_SOURCE=os.environ["HERMES_AGENT_SOURCE"])
    # Use the current interpreter and existing dependency search path; install nothing.
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("host_checkpoint_contract.py"))],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "required failure preserved original evidence" in result.stdout
