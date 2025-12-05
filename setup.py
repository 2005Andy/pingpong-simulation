"""Setup script for PingPong Simulation package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pingpong-simulation",
    version="1.0.0",
    author="PingPong Simulation Team",
    author_email="",
    description="3D ping-pong ball flight simulation with realistic physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/pingpong-simulation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="pingpong simulation physics aerodynamics collision-detection",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "sphinx>=4.0",
        ],
        "animation": [
            "ffmpeg-python>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pingpong-sim=pingpong_main:main",
            "pingpong-analyze=analyze_impact:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
