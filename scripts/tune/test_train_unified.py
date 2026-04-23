#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train  # noqa: E402


class TestUtilityFunctions(unittest.TestCase):
    def test_strip_passthrough(self) -> None:
        self.assertEqual(train.strip_passthrough(["--", "--help"]), ["--help"])
        self.assertEqual(train.strip_passthrough(["--help"]), ["--help"])
        self.assertEqual(train.strip_passthrough([]), [])

    def test_parse_env_pairs(self) -> None:
        env = train.parse_env_pairs(["FOO=bar", "X=1"])
        self.assertEqual(env["FOO"], "bar")
        self.assertEqual(env["X"], "1")

    def test_parse_env_pairs_invalid(self) -> None:
        with self.assertRaises(ValueError):
            train.parse_env_pairs(["BAD"])

    def test_run_subprocess_dry_run(self) -> None:
        ctx = train.RunContext(
            dry_run=True,
            verbose=False,
            workdir=Path(".").resolve(),
            env=dict(),
        )
        result = train.run_subprocess(["echo", "x"], ctx, "task")
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.task, "task")

    def test_run_subprocess_real_build(self) -> None:
        ctx = train.RunContext(
            dry_run=False,
            verbose=False,
            workdir=Path(".").resolve(),
            env=dict(),
        )
        with patch.object(
            train.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=["x"], returncode=0),
        ) as run_mock:
            result = train.run_subprocess(["echo", "x"], ctx, "task")
            self.assertEqual(result.return_code, 0)
            called_cmd = run_mock.call_args.args[0]
            self.assertEqual(called_cmd, ["echo", "x"])


class TestTasks(unittest.TestCase):
    def test_native_python_task_runs(self) -> None:
        called = {"ok": False}

        def _runner(argv):
            called["ok"] = True
            self.assertEqual(list(argv), ["--help"])
            return 0

        task = train.NativePythonTask("esm2", "d", _runner)
        ctx = train.RunContext(
            dry_run=False,
            verbose=False,
            workdir=Path(".").resolve(),
            env=dict(),
        )
        result = task.run(["--help"], ctx)
        self.assertEqual(result.return_code, 0)
        self.assertTrue(called["ok"])

    def test_shell_task_runs(self) -> None:
        task = train.ShellTask(
            "tune-esm2",
            "d",
            Path(train.__file__).resolve().parent / "tune.sh",
            ["--target", "esm2"],
        )
        ctx = train.RunContext(
            dry_run=False,
            verbose=False,
            workdir=Path(".").resolve(),
            env=dict(),
        )
        with patch.object(
            train.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=["x"], returncode=0),
        ) as run_mock:
            result = task.run(["--help"], ctx)
            self.assertEqual(result.return_code, 0)
            called_cmd = run_mock.call_args.args[0]
            self.assertEqual(called_cmd[0], "bash")
            self.assertIn("--target", called_cmd)


class TestManager(unittest.TestCase):
    def test_manager_registers_tasks(self) -> None:
        mgr = train.TrainManager(Path(train.__file__))
        self.assertIn("esm2", mgr.tasks)
        self.assertIn("tune-esm2", mgr.tasks)
        self.assertIn("combo-core-trainers", mgr.tasks)

    def test_run_one_unknown_raises(self) -> None:
        mgr = train.TrainManager(Path(train.__file__))
        ctx = train.RunContext(False, False, Path(".").resolve(), dict())
        with self.assertRaises(KeyError):
            mgr.run_one("not-exist", [], ctx)

    def test_run_chain_stop_on_error(self) -> None:
        mgr = train.TrainManager(Path(train.__file__))
        ctx = train.RunContext(False, False, Path(".").resolve(), dict())
        with patch.object(mgr, "run_one") as run_one:
            run_one.side_effect = [
                train.ExecutionResult("a", ["x"], 0),
                train.ExecutionResult("b", ["x"], 2),
                train.ExecutionResult("c", ["x"], 0),
            ]
            rc = mgr.run_chain(["a", "b", "c"], [], ctx, continue_on_error=False)
            self.assertEqual(rc, 2)
            self.assertEqual(run_one.call_count, 2)

    def test_run_chain_continue_on_error(self) -> None:
        mgr = train.TrainManager(Path(train.__file__))
        ctx = train.RunContext(False, False, Path(".").resolve(), dict())
        with patch.object(mgr, "run_one") as run_one:
            run_one.side_effect = [
                train.ExecutionResult("a", ["x"], 3),
                train.ExecutionResult("b", ["x"], 0),
            ]
            rc = mgr.run_chain(["a", "b"], [], ctx, continue_on_error=True)
            self.assertEqual(rc, 3)
            self.assertEqual(run_one.call_count, 2)


class TestMain(unittest.TestCase):
    def test_main_list(self) -> None:
        with patch.object(train, "TrainManager") as mgr_cls, patch.object(
            sys,
            "argv",
            ["train.py", "list"],
        ):
            mgr = mgr_cls.return_value
            mgr.list_tasks.return_value = [("esm2", "Train ESM2")]
            rc = train.main()
            self.assertEqual(rc, 0)
            mgr.list_tasks.assert_called_once()

    def test_main_run(self) -> None:
        with patch.object(train, "TrainManager") as mgr_cls, patch.object(
            sys,
            "argv",
            ["train.py", "run", "esm2", "--", "--config", "x.json"],
        ):
            mgr = mgr_cls.return_value
            mgr.run_one.return_value = train.ExecutionResult("esm2", ["x"], 0)
            rc = train.main()
            self.assertEqual(rc, 0)
            call = mgr.run_one.call_args
            self.assertEqual(call.args[0], "esm2")
            self.assertEqual(call.args[1], ["--config", "x.json"])

    def test_main_chain(self) -> None:
        with patch.object(train, "TrainManager") as mgr_cls, patch.object(
            sys,
            "argv",
            ["train.py", "chain", "--targets", "esm2,evo2", "--", "--config", "x.json"],
        ):
            mgr = mgr_cls.return_value
            mgr.run_chain.return_value = 5
            rc = train.main()
            self.assertEqual(rc, 5)
            call = mgr.run_chain.call_args
            self.assertEqual(call.args[0], ["esm2", "evo2"])
            self.assertEqual(call.args[1], ["--config", "x.json"])


if __name__ == "__main__":
    unittest.main()
