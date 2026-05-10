import json
import re

# Counters for old checks
double_space_count = 0
tab_count = 0
triple_newline_count = 0

# Counters for new clean_text checks
unicode_char_count = 0
hyphen_break_count = 0
page_num_count = 0
weird_bullet_count = 0

total = 0

# Dictionary to hold one sample context for each issue
samples = {
    'double_space': None,
    'unicode_char': None,
    'hyphen_break': None,
    'page_num': None,
    'weird_bullet': None
}

with open('data/sft_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        total += 1
        obj = json.loads(line)
        for msg in obj['conversations']:
            text = msg['content']
            
            # Skip system prompt
            if msg['role'] == 'system':
                continue

            # Double spaces
            if '  ' in text:
                double_space_count += 1
                if samples['double_space'] is None:
                    m = re.search(r'.{0,30}  .{0,30}', text)
                    if m: samples['double_space'] = repr(m.group())

            # Tabs
            if '\t' in text:
                tab_count += 1

            # Triple Newlines
            if '\n\n\n' in text:
                triple_newline_count += 1

            # Unicode spaces/zero-width chars (\xa0, \u200b)
            if '\xa0' in text or '\u200b' in text:
                unicode_char_count += 1
                if samples['unicode_char'] is None:
                    m = re.search(r'.{0,30}(?:\xa0|\u200b).{0,30}', text)
                    if m: samples['unicode_char'] = repr(m.group())

            # Hyphenated line breaks (e.g., "argu-\nment")
            if re.search(r'[a-zA-Z]+-\s*\n\s*[a-zA-Z]+', text):
                hyphen_break_count += 1
                if samples['hyphen_break'] is None:
                    m = re.search(r'.{0,30}[a-zA-Z]+-\s*\n\s*[a-zA-Z]+.{0,30}', text)
                    if m: samples['hyphen_break'] = repr(m.group())

            # Standalone page numbers on their own line
            if re.search(r'(?m)^\s*\d+\s*$', text):
                page_num_count += 1
                if samples['page_num'] is None:
                    m = re.search(r'(?m)^.{0,10}\s*\d+\s*.{0,10}$', text)
                    if m: samples['page_num'] = repr(m.group())

            # Weird bullet points (\u2022, \u2023, \u25E6, \u2043, \u2219)
            if re.search(r'[\u2022\u2023\u25E6\u2043\u2219]', text):
                weird_bullet_count += 1
                if samples['weird_bullet'] is None:
                    m = re.search(r'.{0,30}[\u2022\u2023\u25E6\u2043\u2219].{0,30}', text)
                    if m: samples['weird_bullet'] = repr(m.group())

print(f"Checked {total} examples for artifacts ")
print("\n")
print(f"Double spaces:           {double_space_count}")
print(f"Tabs:                    {tab_count}")
print(f"Triple+ newlines:        {triple_newline_count}")
print(f"Unicode spaces/chars:    {unicode_char_count}")
print(f"Hyphenated line breaks:  {hyphen_break_count}")
print(f"Standalone page nums:    {page_num_count}")
print(f"Weird bullet points:     {weird_bullet_count}")
print("\n")
print(" SAMPLES FOUND:")

for key, sample in samples.items():
    if sample:
        print(f"{key}:\n  {sample}\n")