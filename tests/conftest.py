import os
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_docs_dir():
    """创建测试文档目录"""
    docs_dir = Path("./docs")
    docs_dir.mkdir(exist_ok=True)
    return docs_dir

@pytest.fixture(scope="session")
def test_data_dir():
    """创建测试数据目录"""
    data_dir = Path("./tests/data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture(scope="session")
def test_output_dir():
    """创建测试输出目录"""
    output_dir = Path("./tests/output")
    output_dir.mkdir(exist_ok=True)
    return output_dir 