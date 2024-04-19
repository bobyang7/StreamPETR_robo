.PHONY: clean deepclean dev

# Use pipenv when not in CI environment and pipenv command exists.
PIPRUN := $(shell [ "${CI}" != "true" ] && command -v pipenv > /dev/null 2>&1 && echo pipenv run)

# Remove common intermediate files.
clean:
	find . -name '*.pyc' -print0 | xargs -0 rm -f
	find . -name '*.swp' -print0 | xargs -0 rm -f
	find . -name '.DS_Store' -print0 | xargs -0 rm -rf
	find . -name '__pycache__' -print0 | xargs -0 rm -rf
	-rm -rf \
		*.egg-info \
		.coverage \
		.eggs \
		.mypy_cache \
		.pytest_cache \
		Pipfile* \
		build \
		dist \
		output \
		public

# Remove common intermediate files alongside with `pre-commit` hook and virtualenv created by `pipenv`.
deepclean: clean
	-pipenv --venv >/dev/null 2>&1 && pipenv --rm

# Prepare virtualenv.
# - Create virtual environment with pipenv and conda python when
#   - Not in CI environment.
#   - No existing venv.
venv:
	-[ "${CI}" != "true" ] && ! pipenv --venv >/dev/null 2>&1 && pipenv --python=/opt/conda/bin/python

# Prepare dev environments:
# - Install package in editable mode alongside with dev requirements.
dev: venv
	${PIPRUN} pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html -c env/constraint.txt
	${PIPRUN} pip install mmdet==2.28.2 -c env/constraint.txt
	${PIPRUN} pip install https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/mmcv_full-1.6.0-cp38-cp38-manylinux1_x86_64.whl
	${PIPRUN} pip install mmsegmentation==0.30.0 -c env/constraint.txt
	${PIPRUN} pip install -e mmdetection3d -c env/constraint.txt
	${PIPRUN} pip install flash-attn==0.2.2 --no-build-isolation --no-cache-dir
	${PIPRUN} pip install fvcore
	${PIPRUN} pip install yapf==0.40.1
	