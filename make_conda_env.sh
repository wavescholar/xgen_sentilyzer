cle
nvidia-smi
mkdir logs

export CONDA_ENV=wavelang_env
conda deactivate
conda info --envs
conda env remove -n $CONDA_ENV
conda env create -f environment.yml
conda activate $CONDA_ENV
rm requirements.txt

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip-compile ./requirements.in  > ./logs/log_requirements-compile.log 2>&1
pip install -r ./requirements.txt  > ./logs/log_requirements-install.log 2>&1

pip install  --use-pep517  google-generativeai
pip install  --use-pep517  google-cloud-secret-manager

echo '42'

