"""Deploy this repository to a Hugging Face Space (Docker SDK).

Called by .github/workflows/deploy.yml so every deployment is a recorded GitHub Actions run whose commit
message names the source commit.  Needs HF_TOKEN (write) and HF_SPACE ("user/space-name").
"""
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ALLOW = ["Dockerfile", ".dockerignore", "requirements.txt", "LICENSE", "swarmsight/**", "web/**", "models/**", "demo/**",
         "docs/**", "datasets/tashkent/**", "benchmarks/results/**"]

SPACE_README = """---
title: SwarmSight
emoji: 🛰️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Collective vision for connected mobility
---

Deployed automatically from {repo} at commit {sha}. Source, README and data licences: https://github.com/{repo}
"""


def main():
    token, space = os.environ["HF_TOKEN"], os.environ["HF_SPACE"]
    sha = os.environ.get("GITHUB_SHA", "local")[:12]
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=".", repo_id=space, repo_type="space", allow_patterns=ALLOW,
                      delete_patterns=["swarmsight/**", "web/**", "models/**", "demo/**", "datasets/**", "benchmarks/**", "docs/**"],
                      commit_message=f"Deploy {repo}@{sha}")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "README.md"
        p.write_text(SPACE_README.format(repo=repo, sha=sha))
        api.upload_file(path_or_fileobj=str(p), path_in_repo="README.md", repo_id=space, repo_type="space",
                        commit_message=f"Space card for {repo}@{sha}")
    print(f"deployed {repo}@{sha} to https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
