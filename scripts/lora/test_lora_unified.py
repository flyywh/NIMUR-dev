#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import scripts.lora.lora as lora  # noqa: E402
except Exception as exc:  # pragma: no cover - environment dependent
    lora = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(lora is None, f"Cannot import scripts.lora.lora: {_IMPORT_ERROR}")
class TestUnifiedDispatch(unittest.TestCase):
    def test_main_dispatch_esm2(self) -> None:
        args = SimpleNamespace(model="esm2")
        with patch.object(lora, "parse_args", return_value=args), patch.object(
            lora, "run_esm2", return_value=11
        ) as esm2_mock, patch.object(lora, "run_evo2") as evo2_mock, patch.object(
            lora, "run_oracle"
        ) as oracle_mock:
            rc = lora.main()
            self.assertEqual(rc, 11)
            esm2_mock.assert_called_once_with(args)
            evo2_mock.assert_not_called()
            oracle_mock.assert_not_called()

    def test_main_dispatch_evo2(self) -> None:
        args = SimpleNamespace(model="evo2")
        with patch.object(lora, "parse_args", return_value=args), patch.object(
            lora, "run_evo2", return_value=22
        ) as evo2_mock:
            rc = lora.main()
            self.assertEqual(rc, 22)
            evo2_mock.assert_called_once_with(args)

    def test_main_dispatch_oracle(self) -> None:
        args = SimpleNamespace(model="oracle")
        with patch.object(lora, "parse_args", return_value=args), patch.object(
            lora, "run_oracle", return_value=33
        ) as oracle_mock:
            rc = lora.main()
            self.assertEqual(rc, 33)
            oracle_mock.assert_called_once_with(args)

    def test_main_dispatch_oracle_cnn(self) -> None:
        args = SimpleNamespace(model="oracle-cnn")
        with patch.object(lora, "parse_args", return_value=args), patch.object(
            lora, "run_oracle_cnn", return_value=44
        ) as oracle_cnn_mock:
            rc = lora.main()
            self.assertEqual(rc, 44)
            oracle_cnn_mock.assert_called_once_with(args)

    def test_main_dispatch_kd_fusion(self) -> None:
        args = SimpleNamespace(model="kd-fusion")
        with patch.object(lora, "parse_args", return_value=args), patch.object(
            lora, "run_kd_fusion", return_value=55
        ) as kd_mock:
            rc = lora.main()
            self.assertEqual(rc, 55)
            kd_mock.assert_called_once_with(args)


@unittest.skipIf(lora is None, f"Cannot import scripts.lora.lora: {_IMPORT_ERROR}")
class TestRequiredArgValidation(unittest.TestCase):
    def test_run_esm2_requires_core_args(self) -> None:
        args = SimpleNamespace(config="", dataset_dir="", test_dataset_path="", output_dir="")
        with self.assertRaises(ValueError):
            lora.run_esm2(args)

    def test_run_evo2_requires_core_args(self) -> None:
        args = SimpleNamespace(config="", train_dataset="", test_dataset="", output_dir="")
        with self.assertRaises(ValueError):
            lora.run_evo2(args)

    def test_run_oracle_requires_core_args(self) -> None:
        args = SimpleNamespace(train_dir="", val_dir="", protbert_path="", output_dir="")
        with self.assertRaises(ValueError):
            lora.run_oracle(args)

    def test_run_oracle_cnn_requires_core_args(self) -> None:
        args = SimpleNamespace(config="", train_dataset="", valid_dataset="", output_dir="")
        with self.assertRaises(ValueError):
            lora.run_oracle_cnn(args)

    def test_run_kd_fusion_requires_core_args(self) -> None:
        args = SimpleNamespace(config="", dataset="", output_dir="")
        with self.assertRaises(ValueError):
            lora.run_kd_fusion(args)


class TestLegacyWrappers(unittest.TestCase):
    def _run_wrapper_and_capture_argv(self, wrapper_name: str, argv_tail: list[str]) -> list[str]:
        captured: list[str] = []
        wrapper_path = THIS_DIR / wrapper_name

        def _fake_main() -> int:
            captured.extend(sys.argv[1:])
            return 0

        with patch("scripts.lora.lora.main", side_effect=_fake_main), patch.object(
            sys, "argv", [wrapper_name, *argv_tail]
        ):
            with self.assertRaises(SystemExit) as cm:
                runpy.run_path(str(wrapper_path), run_name="__main__")
            self.assertEqual(cm.exception.code, 0)
        return captured

    def test_esm2_wrapper_injects_model(self) -> None:
        argv = self._run_wrapper_and_capture_argv("esm2_train_lora.py", ["--config", "x.json"])
        self.assertGreaterEqual(len(argv), 2)
        self.assertEqual(argv[0], "--model")
        self.assertEqual(argv[1], "esm2")
        self.assertIn("--config", argv)

    def test_evo2_wrapper_injects_model(self) -> None:
        argv = self._run_wrapper_and_capture_argv("evo2_train_lora.py", ["--config", "x.json"])
        self.assertGreaterEqual(len(argv), 2)
        self.assertEqual(argv[0], "--model")
        self.assertEqual(argv[1], "evo2")

    def test_oracle_wrapper_injects_model(self) -> None:
        argv = self._run_wrapper_and_capture_argv("oracle_train_lora.py", ["--train_dir", "x"])
        self.assertGreaterEqual(len(argv), 2)
        self.assertEqual(argv[0], "--model")
        self.assertEqual(argv[1], "oracle")

    def test_oracle_cnn_wrapper_injects_model(self) -> None:
        argv = self._run_wrapper_and_capture_argv("oracle_cnn_train_lora.py", ["--config", "x.json"])
        self.assertGreaterEqual(len(argv), 2)
        self.assertEqual(argv[0], "--model")
        self.assertEqual(argv[1], "oracle-cnn")

    def test_kd_fusion_wrapper_injects_model(self) -> None:
        argv = self._run_wrapper_and_capture_argv("kd_fusion_train_lora.py", ["--config", "x.json"])
        self.assertGreaterEqual(len(argv), 2)
        self.assertEqual(argv[0], "--model")
        self.assertEqual(argv[1], "kd-fusion")

    def test_wrapper_keeps_existing_model(self) -> None:
        argv = self._run_wrapper_and_capture_argv(
            "esm2_train_lora.py",
            ["--model", "oracle", "--train_dir", "x"],
        )
        self.assertEqual(argv[0], "--model")
        self.assertEqual(argv[1], "oracle")


if __name__ == "__main__":
    unittest.main()
