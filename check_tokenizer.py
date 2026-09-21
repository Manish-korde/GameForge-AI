from transformers import AutoTokenizer

MODEL_PATH = "models/gameforge_t5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

text = '''{"theme":"Survival","environment":"Dense Wilderness","characters":[{"name":"Stranded Survivor","desc":"A lone survivor trying to endure the wilderness."}],"weapons":[{"name":"Slingshot","desc":"A basic ranged weapon made from scavenged parts."}],"props":[{"name":"Campfire","desc":"A fire used for warmth and cooking."}],"visualStyle":"Stylized 2D"}'''

ids = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(ids)

print("TOKENIZER:", tokenizer.__class__.__name__)
print("VOCAB SIZE:", tokenizer.vocab_size)
print("UNK TOKEN ID:", tokenizer.unk_token_id)
print("UNK TOKEN:", tokenizer.unk_token)

print("\nTOKENS:")
for token_id, token in zip(ids, tokens):
    print(f"{token_id:5d}  {repr(token)}")

print("\nUNK COUNT:", tokens.count(tokenizer.unk_token))
print("TOTAL TOKENS:", len(tokens))