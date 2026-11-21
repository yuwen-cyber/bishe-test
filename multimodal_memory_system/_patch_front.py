from pathlib import Path

p = Path(__file__).resolve().parent / "frontend" / "src" / "api" / "index.ts"
s = p.read_text(encoding="utf-8")
old = "  score: number | null\n}"
new = ("  score: number | null\n"
       "  // 数据来源：formal(正式) / noise(噪声) / user(自录) / unknown\n"
       "  data_source?: 'formal' | 'noise' | 'user' | 'unknown'\n}")
assert old in s, "未找到目标片段"
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("api/index.ts 已加 data_source 字段")
