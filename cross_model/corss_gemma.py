import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
from TokeniserMap import TokenizerMap
HF_TOKEN = "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"
SMALL_MODEL_ID = "google/gemma-2-2b-it"   # pruner
MAIN_MODEL_ID  = "CohereLabs/aya-expanse-8b"   # generator

# optional 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ────────────────────────────────────────────────────────────
# 1.  Gemma-style “weights-only” attention
# ────────────────────────────────────────────────────────────
class GemmaAttentionWeightsOnly(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.q_proj, self.k_proj = original_attn.q_proj, original_attn.k_proj
        self.head_dim   = original_attn.head_dim
        self.num_q      = self.q_proj.out_features // self.head_dim
        self.num_kv     = self.k_proj.out_features // self.head_dim
        self.expand     = self.num_q // self.num_kv
        self.scaling    = getattr(original_attn, "scaling", self.head_dim ** -0.5)
        self.softcap    = getattr(original_attn, "attn_logit_softcapping", None)  # ≈50

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _rope(self, q, k, cos, sin):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )

    def forward(self, hidden, rotary, attn_mask):
        B, L, _ = hidden.shape

        q = self.q_proj(hidden).view(B, L, self.num_q,  self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, L, self.num_kv, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k, rotary[0], rotary[1])

        k = k.repeat_interleave(self.expand, dim=1)                       # GQA
        scores = (q @ k.transpose(-1, -2)) * self.scaling                 # [B,h,L,L]

        if self.softcap is not None:
            scores = torch.tanh(scores / self.softcap) * self.softcap     # Gemma trick

        scores = scores + attn_mask
        return F.softmax(scores, dim=-1)                                  # weights


# ────────────────────────────────────────────────────────────
# 2.  TokenPruner
# ────────────────────────────────────────────────────────────
class TokenPruner(nn.Module):
    def __init__(self, model_id, compression_ratio, device="cuda:0"):
        super().__init__()
        small = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.embeddings        = small.get_input_embeddings()
        self.rotary_embeddings = small.model.rotary_emb
        self.self_attention    = GemmaAttentionWeightsOnly(
            small.model.layers[0].self_attn
        )
        del small
        self.compression_ratio = compression_ratio

    def forward(self, input_ids, attention_mask=None):
        embs = self.embeddings(input_ids)                 # [B,L,D]
        B, L, _ = embs.shape

        pos  = torch.arange(L, device=embs.device).unsqueeze(0)
        rotary = self.rotary_embeddings(embs, position_ids=pos)

        # causal attn mask  (0 on/below diagonal, –inf above)
        m = torch.triu(
            torch.full((L, L), -float("inf"), device=embs.device), 1
        ).unsqueeze(0).unsqueeze(0)

        attn = self.self_attention(embs, rotary, m)       # [B,h,L,L]
        importance = attn.mean(dim=1).mean(dim=1)         # [B,L]

        l = int(L * self.compression_ratio)
        k = max(l, 2)
        _, topk = torch.topk(importance, k, sorted=False, dim=-1)
        topk = torch.sort(topk, dim=-1)[0]

        # keep final token
        need_append = (topk[:, -1] != L - 1)
        last = torch.tensor(L - 1, device=topk.device)
        if need_append:
            last_col = torch.full((B, 1), L - 1, device=topk.device)
            topk = torch.cat([topk, last_col], dim=1)

        pruned_ids = torch.gather(input_ids, 1, topk)
        return pruned_ids, topk


# ────────────────────────────────────────────────────────────
# 3.  GemmaPrunedModel  (same API as your LlamaPrunedModel)
# ────────────────────────────────────────────────────────────
class CrossPrunerModel(nn.Module):
    def __init__(self, main_id, small_id, compression_ratio):
        super().__init__()
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # quantization_config=bnb_config,   # optional
            trust_remote_code=True,
        )
        self.main_tokenizer   = AutoTokenizer.from_pretrained(main_id,  token=HF_TOKEN,
                                                              trust_remote_code=True)
        self.pruner_tokenizer = AutoTokenizer.from_pretrained(small_id, token=HF_TOKEN,
                                                              trust_remote_code=True)

        self.device = self.main_model.device
        self.embeddings = self.main_model.get_input_embeddings()
        self.token_pruner = TokenPruner(small_id, compression_ratio)
        # freeze generator weights
        self.main_model.requires_grad_(False)
        self.tokenizer_map = TokenizerMap(small_id, main_id)
        self.compression_ratio = compression_ratio

    # — helper that runs the pruner —
    def post_tokenizer(self, input_ids,atention_mask=None):
        pruned_tokens_ids,_ = self.token_pruner(input_ids.to("cuda:0"),atention_mask)
        pruned_tokens_ids = pruned_tokens_ids.tolist()
        bos_id = self.pruner_tokenizer.bos_token_id
        for i in range(len(pruned_tokens_ids)):
            if pruned_tokens_ids[i][0] == bos_id:
                pruned_tokens_ids[i] = pruned_tokens_ids[i][1:]
        pruned_tokens = self.pruner_tokenizer.batch_decode(pruned_tokens_ids, skip_special_tokens=False)
        for i in range(len(pruned_tokens)):
            pruned_tokens[i] = self.tokenizer_map.map_string(pruned_tokens[i])
        return pruned_tokens
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model.forward(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens = self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.forward(**self.main_tokenizer(pruned_tokens,return_tensors="pt").to(self.device), **kwargs)

        return output
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model.forward(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens= self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.generate(**self.main_tokenizer(pruned_tokens,return_tensors="pt").to(self.device), **kwargs)
        return output


if __name__ == "__main__":
    main_model = CrossPrunerModel(MAIN_MODEL_ID, SMALL_MODEL_ID, 0.9)
    print(main_model.main_tokenizer.decode(main_model.generate(main_model.pruner_tokenizer('<start_of_turn>user\nThis question refers to the following information.\nI come not to urge personal claims, nor to seek individual benefits; I appear as the advocate of those who cannot plead their own cause; I come as the friend of those who are deserted, oppressed, and desolate. In the Providence of God, I am the voice of the maniac whose piercing cries from the dreary dungeons of your jails penetrate not your Halls of Legislation. I am the Hope of the poor crazed beings who pine in the cells, and stalls, and cages, and waste rooms of your poor-houses. I am the Revelation of hundreds of wailing, suffering creatures, hidden in your private dwellings, and in pens and cabins—shut out, cut off from all healing influences, from all mind-restoring cares.… Could their melancholy histories be spread before you as revealed to my grieved spirit during the last three months, how promptly, how earnestly would you search out the most approved means of relief; how trifling, how insignificant, by comparison, would appear the sacrifices you are asked to make; how would a few dimes and dollars, gathered from each citizen, diminish in value as a possession, compared with the certain benefits and vast good to be secured for the suffering insane...by the consecration and application of a sufficient fund to the construction of a suitable hospital.…\n—Dorothea Dix, Memorial Soliciting a State Hospital for the Protection and Cure of the Insane,\nSubmitted to the General Assembly of North Carolina, November 1848\nDorothea Dix can best be compared to whom?\nA. Abigail Adams\nB. Clara Barton\nC. Shirley Temple\nD. Hillary Clinton\nAnswer: B\n\nThis question refers to the following information.\nRead the following excerpt.\nThe revolutionary seed had penetrated into every country and spread more or less. It was greatly developed under the régime of the military despotism of Bonaparte. His conquests displaced a number of laws, institutions, and customs; broke through bonds sacred among all nations, strong enough to resist time itself; which is more than can be said of certain benefits conferred by these innovators.\nThe monarchs will fulfil the duties imposed upon them by Him who, by entrusting them with power, has charged them to watch over the maintenance of justice, and the rights of all, to avoid the paths of error, and tread firmly in the way of truth. Placed beyond the passions which agitate society, it is in days of trial chiefly that they are called upon to despoil realities of their false appearances, and to show themselves as they are, fathers invested with the authority belonging by right to the heads of families, to prove that, in days of mourning, they know how to be just, wise, and therefore strong, and that they will not abandon the people whom they ought to govern to be the sport of factions, to error and its consequences, which must involve the loss of society.\nUnion between the monarchs is the basis of the policy which must now be followed to save society from total ruin. . . .\nLet them not confound concessions made to parties with the good they ought to do for their people, in modifying, according to their recognized needs, such branches of the administration as require it.\nLet them be just, but strong; beneficent, but strict.\nLet them maintain religious principles in all their purity, and not allow the faith to be attacked and morality interpreted according to the social contract or the visions of foolish sectarians.\nLet them suppress Secret Societies; that gangrene of society.\n—Klemens von Metternich, Political Confession of Faith, 1820\nWhich of the following was the greatest cause of the fears expressed by Metternich in the document above?\nA. The ideas of personal liberty and nationalism conceived during the Enlightenment resulted in radical revolutions that could spread throughout Europe.\nB. The conquest of Europe by Napoleon led to the creation of new factions and shifted the European balance of power.\nC. The power of monarchs had grown to the point where it needed to be checked by other powers within each nation or domination of civilians would occur.\nD. The rising and falling economic cycle of the newly emerging capitalist economy could lead to civilian unrest that must be suppressed.\nAnswer: A\n\nWhich word best summarizes Weber\’s explanation of the development of formally rational law?\nA. Authority.\nB. Charisma.\nC. Co-operation.\nD. Capitalism.\nAnswer: D\n\nA son owed a creditor $5,000. The son\’s father contacted the creditor and told him that he wanted to pay the son\’s debt. The father signed a document that stated the father would pay the son\’s debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?\nA. The father\’s promise and the creditor\’s reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. \nB. Because it was foreseeable that the father\’s promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father\’s promise. \nC. The father\’s five payments to the creditor totaling $2,500 manifested a serious intent on the father\‘s part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. \nD. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. \nAnswer: A\n\nThis question refers to the following information.\n”In the new Code of Laws which I suppose it will be necessary for you to make I desire you would Remember the Ladies, and be more generous and favorable to them than your ancestors. Do not put such unlimited power into the hands of the Husbands. Remember all Men would be tyrants if they could. If particular care and attention is not paid to the Ladies we are determined to foment a Rebellion, and will not hold ourselves bound by any Laws in which we have no voice, or Representation.“\nAbigail Adams, in a letter to John Adams, 1776\n”Special legislation for woman has placed us in a most anomalous position. Women invested with the rights of citizens in one section—voters, jurors, office-holders—crossing an imaginary line, are subjects in the next. In some States, a married woman may hold property and transact business in her own name; in others, her earnings belong to her husband. In some States, a woman may testify against her husband, sue and be sued in the courts; in others, she has no redress in case of damage to person, property, or character. In case of divorce on account of adultery in the husband, the innocent wife is held to possess no right to children or property, unless by special decree of the court. But in no State of the Union has the wife the right to her own person, or to any part of the joint earnings of the co-partnership during the life of her husband. In some States women may enter the law schools and practice in the courts; in others they are forbidden. In some universities girls enjoy equal educational advantages with boys, while many of the proudest institutions in the land deny them admittance, though the sons of China, Japan and Africa are welcomed there. But the privileges already granted in the several States are by no means secure.“\nSusan B. Anthony, “Declaration of Rights for Women,” July 4, 1876\nThe sentiments expressed in the second excerpt by Susan B. Anthony are most likely in support of\nA. the Equal Rights Amendment\nB. universal suffrage\nC. states\’ rights\nD. prohibition\nAnswer: B\n\nThis question refers to the following information.\nPerestroika [Restructuring] is an urgent necessity arising from the profound processes of development in our socialist society. This society is ripe for change. It has long been yearning for it. Any delay in beginning perestroika could have led to an exacerbated internal situation in the near future, which, to put it bluntly, would have been fraught with serious social, economic, and political crises.\nMikhail Gorbachev, Perestroika: New Thinking for Our Country and the World, 1987\nFrom the passage, one may infer that Gorbachev believed that\nA. the problems that required perestroika were the fault of capitalist enemies of socialism\nB. the problems that required perestroika were internal to the development of socialist society\nC. a socialist society could not work\nD. a socialist society could not coexist with capitalism\nAnswer:<end_of_turn>\n<start_of_turn>model',return_tensors="pt")["input_ids"])[0]))