from pathlib import Path
import pytest
from sme.config import SmeConfig


def test_smeconfig():
    """Test code paths and case in config.SmeConfig().
    """
    default_path = Path.home() / '.sme' / 'sme.cfg'
    cfg = SmeConfig()
    assert cfg.path == default_path

    for custom_path in ['sme.cfg', '~/sme.cfg', '/sme.cfg']:
        cfg = SmeConfig(custom_path)
        assert cfg.path == Path(custom_path).expanduser()

    cfg = SmeConfig('')
    assert isinstance(cfg, SmeConfig)
