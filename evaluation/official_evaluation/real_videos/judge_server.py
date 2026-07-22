#!/usr/bin/env python3
"""轻量 OpenAI 兼容 judge server，加载本地 Qwen3-8B。"""
import argparse
import json
import sys
import torch
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)
model = None
tokenizer = None


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    user_content = messages[-1]["content"] if messages else ""
    system_content = messages[0]["content"] if len(messages) > 1 else ""

    prompt = f"{system_content}\n\n{user_content}" if system_content else user_content

    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": "你是一个评分助手。请直接输出JSON，不要思考。"},
         {"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    resp_json = {
        "choices": [{"message": {"content": response}}],
        "model": "qwen3-8b-local",
    }
    return Response(json.dumps(resp_json, ensure_ascii=False), mimetype="application/json")


@app.route("/health")
def health():
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/caiqingyuan/code/lifebench/models/Qwen__Qwen3-8B")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map={"": args.gpu},
    )
    model.eval()
    print(f"[GPU {args.gpu}] Model loaded on port {args.port}")

    from waitress import serve
    serve(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
