from pathlib import Path

p = Path(r"C:\Users\sjdin\CC\reelmotion-ai\.venv\lib\site-packages\safetensors\torch.py")
text = p.read_text(encoding="utf-8")
old = 'backend: str = "mmap",'
new = 'backend: str = "pread" if os.name == "nt" else "mmap",'
if old in text:
    text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")
    print("Patched safetensors torch.py successfully!")
else:
    print("Already patched or pattern not found.")
