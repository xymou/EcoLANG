export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server --model /remote-home/share/LLM_CKPT/huggingface_models/Llama-3.1-8B-Instruct/ --served-model-name llama-3 --dtype half --tool-call-parser None --max-model-len 23104 --enable_chunked_prefill False --port 8001 --logits-processor-pattern ".*Vocab.*"
