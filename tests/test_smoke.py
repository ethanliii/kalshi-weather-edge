import weather_edge
from weather_edge import config


def test_package_imports():
    assert weather_edge.__doc__
    assert config.ROOT.joinpath("pyproject.toml").exists()
