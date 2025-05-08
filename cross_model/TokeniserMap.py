from transformers import AutoTokenizer
class TokenizerMap():
    def __init__(self, pruner_tokenizer, main_tokenizer, token = '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'):
        self.pruner_tokenizer = AutoTokenizer.from_pretrained(pruner_tokenizer, trust_remote_code=True, token=token)
        self.main_tokenizer = AutoTokenizer.from_pretrained(main_tokenizer, trust_remote_code=True, token=token)
        self.map_dict = self._create_tokenizer_map()
        print(self.map_dict)

    def _create_tokenizer_map(self):
        map_dict = {}
        for token_type in self.pruner_tokenizer.special_tokens_map:
            if token_type in self.main_tokenizer.special_tokens_map:
                map_dict[self.pruner_tokenizer.special_tokens_map[token_type]] = self.main_tokenizer.special_tokens_map[token_type]
        return map_dict
    
    def map_string(self, string):
        for i in self.map_dict:
            while i in string:
                string = string.replace(i, self.map_dict[i])
                print(string)
        return string

