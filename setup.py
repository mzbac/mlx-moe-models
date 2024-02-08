from pathlib import Path
from setuptools import setup
import pkg_resources


with open(Path(__file__).parent / "requirements.txt") as fid:
    requirements = [str(r) for r in pkg_resources.parse_requirements(fid)]

setup(
    name="mlx-moe",
    version="0.0.1",
    description="A tool to generate text with mlx-moe model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/yourproject",
    license="MIT",
    install_requires=requirements,
    packages=[
        "mlx_moe",
    ],
    python_requires=">=3.8",
)
