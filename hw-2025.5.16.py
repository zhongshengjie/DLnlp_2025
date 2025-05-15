print(1)
#!/usr/bin/env python
# auto_webpage.py
# 生成静态 HTML 着陆页（本地 LLM 推理）

import torch, subprocess, webbrowser, textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"   # 也可换成 DeepSeek-LLaMA-7B 等
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME,
                                        local_files_only=False,
                                        resume_download=True,
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",             # 单卡 3090 会自动放到 GPU0
    )
    return tok, model

def build_prompt(topic, style):
    sys_inst = (
        "You are a senior web designer. "
        "Generate a **complete, valid HTML5 landing page** with inline TailwindCSS CDN "
        "and minimal JS. No markdown, no commentary – output HTML only."
    )
    user_inst = f"Theme: {topic}\nTone: {style}\nSections: hero banner, features (3), call-to-action, footer."
    return f"<|system|>\n{sys_inst}\n<|user|>\n{user_inst}\n<|assistant|>\n"

def generate_page(tokenizer, model, prompt, max_new=7000):
    in_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_ids = model.generate(
        **in_ids,
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        streamer=TextStreamer(tokenizer)  # 实时打印生成进度
    )
    html = tokenizer.decode(gen_ids[0][in_ids["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return html.strip()

def main():
    topic = input("🔤 请输入网页主题，例如『AI 编程助手』: ")
    style = input("🎨 请输入整体基调，例如『清新极简』: ")
    tokenizer, model = load_model()
    html = generate_page(tokenizer, model, build_prompt(topic, style))
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ 生成完毕 → index.html")
    # 使用系统默认浏览器打开
    webbrowser.open("file://" + subprocess.check_output(["pwd"]).decode().strip() + "/index.html")

if __name__ == "__main__":
    main()
