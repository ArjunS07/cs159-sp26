from __future__ import annotations

import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from pnp import env_setup


class QuantizationCompatTests(unittest.TestCase):
    def test_all_constants_are_patched_idempotently(self):
        quantization = types.ModuleType("torch.ao.quantization")
        with mock.patch.object(env_setup.importlib, "import_module", return_value=quantization):
            self.assertEqual(
                env_setup._fix_torch_quant_compat(),
                ("CUSTOM_KEY", "NUMERIC_DEBUG_HANDLE_KEY", "FROM_NODE_KEY"),
            )
            self.assertEqual(env_setup._fix_torch_quant_compat(), ())
        self.assertEqual(quantization.CUSTOM_KEY, "custom")
        self.assertEqual(quantization.NUMERIC_DEBUG_HANDLE_KEY, "numeric_debug_handle")
        self.assertEqual(quantization.FROM_NODE_KEY, "from_node")


class OptionalPackageTests(unittest.TestCase):
    @mock.patch.object(env_setup, "_uninstall_optional")
    @mock.patch.object(env_setup.importlib.util, "find_spec", return_value=None)
    def test_missing_torchaudio_is_ignored(self, _find_spec, uninstall):
        self.assertFalse(env_setup._remove_broken_optional_torchaudio())
        uninstall.assert_not_called()

    @mock.patch.object(env_setup, "_uninstall_optional")
    @mock.patch.object(env_setup.importlib, "import_module")
    @mock.patch.object(env_setup.importlib.util, "find_spec", return_value=object())
    def test_working_torchaudio_is_retained(self, _find_spec, import_module, uninstall):
        self.assertFalse(env_setup._remove_broken_optional_torchaudio())
        import_module.assert_called_once_with("torchaudio")
        uninstall.assert_not_called()

    @mock.patch.object(env_setup, "_uninstall_optional")
    @mock.patch.object(env_setup.importlib, "import_module", side_effect=OSError("bad CUDA wheel"))
    @mock.patch.object(env_setup.importlib.util, "find_spec", return_value=object())
    def test_broken_torchaudio_is_removed(self, _find_spec, _import_module, uninstall):
        self.assertTrue(env_setup._remove_broken_optional_torchaudio())
        uninstall.assert_called_once_with("torchaudio")

    @mock.patch.object(env_setup, "_uninstall_optional")
    def test_torchao_is_removed_only_when_installed(self, uninstall):
        with mock.patch.object(env_setup.importlib.util, "find_spec", return_value=None):
            self.assertFalse(env_setup._remove_torchao())
        with mock.patch.object(env_setup.importlib.util, "find_spec", return_value=object()):
            self.assertTrue(env_setup._remove_torchao())
        uninstall.assert_called_once_with("torchao")


class RuntimeValidationTests(unittest.TestCase):
    def test_core_incompatibility_is_actionable_and_does_not_install(self):
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with mock.patch.object(env_setup.importlib, "import_module", return_value=torch), \
                mock.patch.object(env_setup.subprocess, "check_call") as check_call:
            with self.assertRaisesRegex(RuntimeError, "fresh GPU runtime"):
                env_setup._require_core_runtime()
        check_call.assert_not_called()

    def test_missing_credentials_are_actionable(self):
        hub = types.ModuleType("huggingface_hub")
        hub.get_token = lambda: None
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN Colab secret"):
                env_setup._require_hf_credentials()

    def test_existing_huggingface_token_is_accepted_without_login(self):
        hub = types.ModuleType("huggingface_hub")
        hub.get_token = lambda: "stored-token"
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(env_setup._require_hf_credentials(), "stored-token")

    def test_model_import_failure_never_prints_ready(self):
        torch = types.SimpleNamespace(
            __version__="2.test",
            version=types.SimpleNamespace(cuda="13.test"),
            cuda=types.SimpleNamespace(get_device_name=lambda _index: "test GPU"),
        )
        torchvision = types.SimpleNamespace(__version__="0.test")
        output = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(env_setup, "_require_core_runtime", return_value=(torch, torchvision)), \
                mock.patch.object(env_setup, "_fix_torch_quant_compat", return_value=()), \
                mock.patch.object(env_setup, "_remove_torchao"), \
                mock.patch.object(env_setup, "_remove_broken_optional_torchaudio"), \
                mock.patch.object(env_setup, "_require_hf_credentials", return_value="token"), \
                mock.patch.object(env_setup, "_verify_model_and_sim_imports", side_effect=RuntimeError("bad LeRobot")), \
                redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "bad LeRobot"):
                env_setup.setup_environment(hf_home="/content/local-hf")
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")
            self.assertEqual(os.environ["HF_HOME"], "/content/local-hf")
        self.assertNotIn("Environment ready.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
