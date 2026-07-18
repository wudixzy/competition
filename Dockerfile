FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

ENV PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:${PATH}
ENV PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
ENV VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 BI100_EXECUTOR_STARTUP_DEBUG=1 ENABLE_CUSTOM_IPC=1

RUN mkdir /workspace
WORKDIR /workspace/
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./vllm/core/block/prefix_caching_block.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/prefix_caching_block.py
COPY ./vllm/core/block/block_table.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/block_table.py
COPY ./vllm/core/block_manager_v2.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block_manager_v2.py
RUN cd ./qwen3_6_scripts && bash ./patch_ops.sh
