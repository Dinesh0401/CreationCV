from pathlib import Path

p = Path(r"C:\Users\sjdin\CC\reelmotion-ai\.venv\lib\site-packages\safetensors\torch.py")
text = p.read_text(encoding="utf-8")

old_block = '''    with safe_open(filename, framework="pt", device=device, backend=backend) as f:
        return f.get_tensors()'''

new_block = '''    if isinstance(device, torch.device):
        device = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"
    elif device == "cuda":
        device = "cuda:0"
    with safe_open(filename, framework="pt", device=device, backend=backend) as f:
        return f.get_tensors()'''

if old_block in text:
    text = text.replace(old_block, new_block, 1)
    p.write_text(text, encoding="utf-8")
    print("Successfully patched safetensors torch.py with device normalization!")
else:
    print("Old block not found or already patched.")
