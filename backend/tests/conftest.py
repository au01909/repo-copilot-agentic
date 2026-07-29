import shutil

import pytest

from app import chunking, ingest
from app.search import HybridSearchIndex

TEST_REPO_URL = "https://github.com/karpathy/micrograd.git"


@pytest.fixture(scope="session")
def cloned_repo():
    local_dir = ingest.clone_repository(TEST_REPO_URL)
    yield local_dir
    shutil.rmtree(local_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def repo_chunks(cloned_repo):
    files = ingest.walk_repository(cloned_repo)
    chunks = []
    for f in files:
        chunks += chunking.chunk_file(TEST_REPO_URL, f.path, f.language, f.content)
    ingest.enrich_chunks_with_git_metadata(cloned_repo, chunks)
    return chunks


@pytest.fixture(scope="session")
def search_index(repo_chunks):
    idx = HybridSearchIndex(repo_chunks)
    yield idx
    idx.close()
