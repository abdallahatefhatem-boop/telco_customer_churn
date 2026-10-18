.PHONY: creat_venv_ubuntu_activate creat_venv_windows_activate install_req run test dvc_add clean_logs

creat_venv_ubuntu_activate:
	uv venv .venv

creat_venv_windows_activate:
	uv venv .venv

install_req:
	uv sync

run:
	uv run python app.py

test:
	uv run pytest

dvc_add:
	.venv/bin/dvc add artifacts

clean_logs:
	rm -rf logs/*.log