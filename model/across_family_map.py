import re

llama3 = {
    "bos": "<|start_of_text|>",
    "end_of_turn": "<|eot_id|>",
    "system": "<|start_header_id|>system<|end_header_id|>\n\n",
    "user": "<|start_header_id|>user<|end_header_id|>\n\n",
    "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
}

gemma2 = {
    "bos": "<bos>",
    "end_of_turn": "<end_of_turn>\n",
    "user": "<start_of_turn>user\n",
    "assistant": "<start_of_turn>model\n",
    "system": "<start_of_turn>model\n",
}
aya_expanse = {
    "bos": "<BOS_TOKEN>",
    "end_of_turn": "<|END_OF_TURN_TOKEN|>",
    "user": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
    "assistant": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "system": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
}


def convert_gemma2_to_llama3(text):
    # Replace the special tokens in the text
    for key, value in gemma2.items():
        text = re.sub(re.escape(value), llama3[key], text)
    return text


def convert_llama3_to_gemma2(text):
    # Replace the special tokens in the text
    for key, value in llama3.items():
        text = re.sub(re.escape(value), gemma2[key], text)
    return text


def convert_gemma2_to_aya_expanse(text):
    # Replace the special tokens in the text
    for key, value in gemma2.items():
        text = re.sub(re.escape(value), aya_expanse[key], text)
    return text


def convert_aya_expanse_to_gemma2(text):
    # Replace the special tokens in the text
    for key, value in aya_expanse.items():
        text = re.sub(re.escape(value), gemma2[key], text)
    return text


def convert_llama3_to_aya_expanse(text):
    # Replace the special tokens in the text
    for key, value in llama3.items():
        text = re.sub(re.escape(value), aya_expanse[key], text)
    return text


def convert_aya_expanse_to_llama3(text):
    # Replace the special tokens in the text
    for key, value in aya_expanse.items():
        text = re.sub(re.escape(value), llama3[key], text)
    return text


def convert_llama3_to_gemma2(text):
    # Replace the special tokens in the text
    for key, value in llama3.items():
        text = re.sub(re.escape(value), gemma2[key], text)
    return text
