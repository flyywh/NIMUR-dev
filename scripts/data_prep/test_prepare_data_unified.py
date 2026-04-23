#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import adapters  # noqa: E402
import prepare_data  # noqa: E402


class TestAdapters(unittest.TestCase):
    def test_legacy_script_exists(self) -> None:
        path = adapters.legacy_script("01_prepare_esm2_data.sh")
        self.assertTrue(path.exists())
        self.assertTrue(path.name.endswith(".sh"))

    def test_missing_legacy_script_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            adapters.legacy_script("definitely_missing_script.sh")


class TestCommandBuilders(unittest.TestCase):
    def test_cmd_assets(self) -> None:
        args = SimpleNamespace(env="evo2", download_models=["NIMROD-ESM", "ProtBERT"])
        cmd, outputs, inputs, params, legacy = prepare_data.cmd_assets(args)
        self.assertEqual(legacy, "00_prepare_assets.sh")
        self.assertEqual(cmd[0], "bash")
        self.assertIn("--download", cmd)
        self.assertEqual(outputs, [])
        self.assertEqual(inputs, [])
        self.assertEqual(params["env"], "evo2")

    def test_cmd_esm2(self) -> None:
        args = SimpleNamespace(
            env="evo2",
            train_pos_fasta="a.fasta",
            train_neg_fasta="b.fasta",
            valid_pos_fasta="c.fasta",
            valid_neg_fasta="d.fasta",
            test_pos_fasta="e.fasta",
            test_neg_fasta="f.fasta",
            output_dir="out/esm2",
            gpus="0,1",
            mock=True,
            mock_dim=1280,
        )
        cmd, outputs, inputs, _params, legacy = prepare_data.cmd_esm2(args)
        self.assertEqual(legacy, "01_prepare_esm2_data.sh")
        self.assertEqual(cmd[0], "bash")
        self.assertIn("--train-pos", cmd)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(inputs), 6)

    def test_cmd_evo2(self) -> None:
        args = SimpleNamespace(
            env="evo2",
            train_pos_fasta="tp.fna",
            train_neg_fasta="tn.fna",
            test_pos_fasta="vp.fna",
            test_neg_fasta="vn.fna",
            output_dir="out/evo2",
            gpus="",
            layer="blocks.21.mlp.l3",
            max_len=1024,
            report_every=10,
            samples_per_shard=8,
            dtype="float16",
            min_free_gb=1,
            stop_on_low_disk=False,
            mock=False,
            mock_dim=1920,
        )
        cmd, outputs, inputs, _params, legacy = prepare_data.cmd_evo2(args)
        self.assertEqual(legacy, "03_prepare_evo2_data.sh")
        self.assertEqual(cmd[0], "bash")
        self.assertIn("--work-dir", cmd)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(inputs), 4)

    def test_cmd_kg(self) -> None:
        args = SimpleNamespace(
            input_tsv="in.tsv",
            output_dir="out/kg",
            train_ratio=0.8,
            seed=42,
        )
        cmd, outputs, inputs, _params, legacy = prepare_data.cmd_kg(args)
        self.assertEqual(legacy, "05_prepare_kg_data.sh")
        self.assertEqual(cmd[0], "bash")
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(inputs), 1)

    def test_cmd_ee(self) -> None:
        args = SimpleNamespace(
            env="evo2",
            gpus="0",
            evo_emb_root="out/evo2/emb",
            esm_train_dataset="train.pt",
            esm_valid_dataset="valid.pt",
            esm_test_dataset="test.pt",
            output_dir="out/ee",
            allow_missing=True,
        )
        cmd, outputs, inputs, _params, legacy = prepare_data.cmd_ee(args)
        self.assertEqual(legacy, "13_prepare_ee_data.sh")
        self.assertEqual(cmd[0], "bash")
        self.assertIn("--allow-missing", cmd)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(inputs), 3)

    def test_cmd_kd_fusion(self) -> None:
        args = SimpleNamespace(
            evo_emb_root="out/evo2/emb",
            esm_train="a.pt",
            esm_valid="b.pt",
            esm_test="c.pt",
            protbert="local_assets/data/ProtBERT/ProtBERT",
            train_pos_fasta="tp.fna",
            train_neg_fasta="tn.fna",
            test_pos_fasta="vp.fna",
            test_neg_fasta="vn.fna",
            val_ratio=0.2,
            seed=42,
            max_samples=50,
            output_file="out/kd/fused.pt",
        )
        cmd, outputs, inputs, _params, legacy = prepare_data.cmd_kd_fusion(args)
        self.assertEqual(legacy, "14_prepare_kd_fusion_data.py")
        self.assertEqual(cmd[0], "python")
        self.assertIn("--out", cmd)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(inputs), 7)

    def test_cmd_oracle_cnn_v1_v2(self) -> None:
        v1 = SimpleNamespace(
            label_file="label.txt",
            train_fasta="train.fasta",
            valid_fasta="valid.fasta",
            output_dir="out/oracle",
            use_v2=False,
            max_seq_len=1022,
        )
        cmd1, _, _, _, legacy1 = prepare_data.cmd_oracle_cnn(v1)
        self.assertEqual(legacy1, "ORACLE_CNN_dataset.py")
        self.assertNotIn("--max_seq_len", cmd1)

        v2 = SimpleNamespace(**{**v1.__dict__, "use_v2": True})
        cmd2, _, _, _, legacy2 = prepare_data.cmd_oracle_cnn(v2)
        self.assertEqual(legacy2, "ORACLE_CNN_dataset_v2.py")
        self.assertIn("--max_seq_len", cmd2)


class TestRunAndMain(unittest.TestCase):
    def test_run_with_manifest_real_command(self) -> None:
        # Real smoke execution: actually spawn a subprocess via current Python,
        # without touching heavy model/data pipelines.
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "real_smoke"
            cmd = [sys.executable, "-c", "print('smoke-ok')"]
            rc = prepare_data.run_with_manifest(
                pipeline="smoke",
                cmd=cmd,
                output_dir=out_dir,
                outputs=[],
                input_files=[],
                params={"mode": "real_subprocess"},
                adapter_target="direct",
            )
            self.assertEqual(rc, 0)
            manifest_path = out_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["pipeline"], "smoke")
            self.assertEqual(data["return_code"], 0)

    def test_run_with_manifest_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"

            with patch.object(
                prepare_data,
                "run_cmd",
                return_value=subprocess.CompletedProcess(args=["x"], returncode=0),
            ):
                rc = prepare_data.run_with_manifest(
                    pipeline="esm2",
                    cmd=["echo", "ok"],
                    output_dir=out_dir,
                    outputs=[out_dir / "a.pt"],
                    input_files=[Path("in.fasta")],
                    params={"k": "v"},
                    adapter_target="01_prepare_esm2_data.sh",
                )
                self.assertEqual(rc, 0)
                data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(data["status"], "success")
                self.assertEqual(data["pipeline"], "esm2")

            with patch.object(
                prepare_data,
                "run_cmd",
                return_value=subprocess.CompletedProcess(args=["x"], returncode=2),
            ):
                rc = prepare_data.run_with_manifest(
                    pipeline="kg",
                    cmd=["echo", "bad"],
                    output_dir=out_dir,
                    outputs=[],
                    input_files=[],
                    params={},
                    adapter_target="05_prepare_kg_data.sh",
                )
                self.assertEqual(rc, 2)
                data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(data["status"], "failed")

    def test_main_assets_branch(self) -> None:
        fake = (["bash", "x.sh"], [], [], {}, "00_prepare_assets.sh")
        with patch.object(prepare_data, "cmd_assets", return_value=fake), patch.object(
            prepare_data, "run_with_manifest", return_value=0
        ) as run_mock, patch.object(
            sys, "argv", ["prepare_data.py", "assets", "--env", "evo2"]
        ):
            rc = prepare_data.main()
            self.assertEqual(rc, 0)
            run_mock.assert_called_once()
            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["pipeline"], "assets")
            self.assertIn("data_prep_unified_assets_run", str(kwargs["output_dir"]))

    def test_main_kd_fusion_manifest_parent(self) -> None:
        fake = (["python", "x.py"], [Path("out/fused.pt")], [], {}, "14_prepare_kd_fusion_data.py")
        with patch.object(prepare_data, "cmd_kd_fusion", return_value=fake), patch.object(
            prepare_data, "run_with_manifest", return_value=0
        ) as run_mock, patch.object(
            sys,
            "argv",
            [
                "prepare_data.py",
                "kd-fusion",
                "--evo-emb-root",
                "emb",
                "--esm-train",
                "a.pt",
                "--esm-test",
                "b.pt",
                "--train-pos-fasta",
                "tp.fna",
                "--train-neg-fasta",
                "tn.fna",
                "--test-pos-fasta",
                "vp.fna",
                "--test-neg-fasta",
                "vn.fna",
                "--output-file",
                "x/y/fused.pt",
            ],
        ):
            rc = prepare_data.main()
            self.assertEqual(rc, 0)
            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["pipeline"], "kd-fusion")
            self.assertEqual(kwargs["output_dir"], Path("x/y/fused.pt").resolve().parent)

    def test_main_regular_output_dir_branch(self) -> None:
        fake = (["bash", "x.sh"], [Path("out/train.pt")], [], {}, "03_prepare_evo2_data.sh")
        with patch.object(prepare_data, "cmd_evo2", return_value=fake), patch.object(
            prepare_data, "run_with_manifest", return_value=0
        ) as run_mock, patch.object(
            sys,
            "argv",
            [
                "prepare_data.py",
                "evo2",
                "--train-pos-fasta",
                "tp.fna",
                "--train-neg-fasta",
                "tn.fna",
                "--test-pos-fasta",
                "vp.fna",
                "--test-neg-fasta",
                "vn.fna",
                "--output-dir",
                "out/evo2",
            ],
        ):
            rc = prepare_data.main()
            self.assertEqual(rc, 0)
            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["pipeline"], "evo2")
            self.assertIn("out", str(kwargs["output_dir"]))


if __name__ == "__main__":
    unittest.main()
