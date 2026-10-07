import torch
from torch.utils.data import Dataset


def causal_mask(size):
    # 下三角 Mask：当前位置只能看到自己和之前的位置
    return torch.triu(torch.ones(1, size, size), diagonal=1).bool() == 0


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt,
                 src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # 源语言特殊 Token
        self.src_sos = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")]
        )
        self.src_eos = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")]
        )
        self.src_pad = tokenizer_src.token_to_id("[PAD]")

        # 目标语言特殊 Token
        self.tgt_sos = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")]
        )
        self.tgt_eos = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")]
        )
        self.tgt_pad = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        # 返回数据集样本数量
        return len(self.ds)

    def __getitem__(self, idx):
        # 取一组英中翻译数据
        pair = self.ds[idx]["translation"]
        src_text = pair[self.src_lang]
        tgt_text = pair[self.tgt_lang]

        # 文本 -> Token IDs
        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Encoder 需要 [SOS] 和 [EOS]，所以减 2
        enc_pad_num = self.seq_len - len(enc_tokens) - 2

        # Decoder input 只需要 [SOS]，所以减 1
        dec_pad_num = self.seq_len - len(dec_tokens) - 1

        # 超过最大长度直接报错
        if enc_pad_num < 0 or dec_pad_num < 0:
            raise ValueError("Sentence is too long")

        # Encoder Input:
        # [SOS] + source + [EOS] + [PAD]...
        encoder_input = torch.cat([
            self.src_sos,
            torch.tensor(enc_tokens),
            self.src_eos,
            torch.full((enc_pad_num,), self.src_pad)
        ]).long()

        # Decoder Input:
        # [SOS] + target + [PAD]...
        decoder_input = torch.cat([
            self.tgt_sos,
            torch.tensor(dec_tokens),
            torch.full((dec_pad_num,), self.tgt_pad)
        ]).long()

        # Label:
        # target + [EOS] + [PAD]...
        label = torch.cat([
            torch.tensor(dec_tokens),
            self.tgt_eos,
            torch.full((dec_pad_num,), self.tgt_pad)
        ]).long()

        # 检查长度是否统一为 seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Encoder Mask：屏蔽 PAD
        # (seq_len) -> (1, 1, seq_len)
        encoder_mask = (
            (encoder_input != self.src_pad)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Decoder Mask：
        # 1. 屏蔽 PAD
        # 2. 屏蔽未来位置
        decoder_mask = (
            (decoder_input != self.tgt_pad).unsqueeze(0)
            & causal_mask(self.seq_len)
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    