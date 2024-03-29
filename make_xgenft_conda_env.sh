clear
nvidia-smi
mkdir logs

export CONDA_ENV=xgen_env
conda deactivate
conda info --envs
conda env remove -n $CONDA_ENV
conda env create -f environment.yml
conda activate $CONDA_ENV
rm requirements.txt

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# See https://github.com/huggingface/transformers/issues/28472 we need to use 4.33 or below
pip install transformers[torch]==4.33
pip install xformers
pip install accelerate peft bitsandbytes trl datasets --upgrade
pip install tiktoken

#Feature attribution and visualization tools for PyTorch
conda install -y captum flask-compress matplotlib -c pytorch

pip install jupyter

#bbcrevisit - captum didn't insatll correctly
pip install captum


echo '42'
