FROM python:3.9

# set USER
RUN useradd -m -r -u 1101 newuser  # create user
RUN python3 -m pip install --upgrade pip
USER newuser

RUN python3 -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install pytorch-lightning==2.3.1

RUN python3 -m pip install x-transformers==1.31.6
RUN python3 -m pip install einops numpy matplotlib scikit-learn wandb pandas supervised-fcn-2 numba


# to enter into the command window (instead of python window)
CMD ["/bin/bash"]