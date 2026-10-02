"""Loader API regression: dispatch checkpoint type through the base config."""
import sys
from types import ModuleType, SimpleNamespace


def test_loader_dispatches_config_and_preserves_pinned_processors(monkeypatch):
    from pnp import models
    cfg = SimpleNamespace(vlm_model_name='HuggingFaceTB/SmolVLM2-500M-Instruct',
                          chunk_size=50, num_steps=10, n_action_steps=1)
    calls = []
    class BaseConfig:
        @classmethod
        def from_pretrained(cls, path):
            calls.append(('config', path))
            return cfg
    class Policy:
        config = cfg
        model = SimpleNamespace(config=cfg)
        @classmethod
        def from_pretrained(cls, path, config):
            assert config is cfg
            return cls()
        def to(self, device): return self
        def eval(self): return self
    def processors(config, path, preprocessor_overrides):
        assert config is cfg
        assert preprocessor_overrides['tokenizer_processor']['tokenizer_name'] == '/vlm'
        return 'pre', 'post'
    modules = {
        'lerobot.configs.policies': {'PreTrainedConfig': BaseConfig},
        'lerobot.policies.smolvla.modeling_smolvla': {'SmolVLAPolicy': Policy},
        'lerobot.policies.factory': {'make_pre_post_processors': processors},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    def weights(repo, revision):
        calls.append((repo, revision))
        return '/vlm' if 'SmolVLM2' in repo else '/policy'
    monkeypatch.setattr(models, '_ensure_hf_weights', weights)
    monkeypatch.setattr(models, 'apply_smolvla_pnp_patch', lambda policy: None)
    policy, pre, post = models.load_smolvla(device='cpu')
    assert ('config', '/policy') in calls
    assert policy._pnp_policy_revision == '6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
    assert (pre, post) == ('pre', 'post')
