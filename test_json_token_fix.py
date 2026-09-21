from transformers import AutoTokenizer

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("BEFORE FIX")
print("Vocab length:", len(tokenizer))
print("{ token:", tokenizer.convert_tokens_to_ids("{"))
print("} token:", tokenizer.convert_tokens_to_ids("}"))
print("<unk> id:", tokenizer.unk_token_id)

tokens_to_add = []

if tokenizer.convert_tokens_to_ids("{") == tokenizer.unk_token_id:
    tokens_to_add.append("{")

if tokenizer.convert_tokens_to_ids("}") == tokenizer.unk_token_id:
    tokens_to_add.append("}")

print("\nAdding:", tokens_to_add)

added = tokenizer.add_tokens(tokens_to_add)

print("Added:", added)

print("\nAFTER FIX")
print("Tokenizer length:", len(tokenizer))
print("{ token:", tokenizer.convert_tokens_to_ids("{"))
print("} token:", tokenizer.convert_tokens_to_ids("}"))

text = '{"theme":"Survival","characters":[{"name":"Hero","desc":"A brave hero."}]}'

ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

tokens = tokenizer.convert_ids_to_tokens(ids)

print("\nTOKENS:")
print(list(zip(ids, tokens)))

print("\nUNK COUNT:", tokens.count(tokenizer.unk_token))