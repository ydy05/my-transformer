import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import warnings
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_config, get_weights_file_path

from dataset import BilingualDataset, causal_mask
from transformer import build_transformer

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Encoder 只计算一次
    encoder_output = model.encode(source, source_mask)

    # Decoder 初始输入：[SOS]
    decoder_input = torch.tensor([[sos_idx]], dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:

        # 屏蔽未来 Token
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        # Decoder 输出
        decoder_output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # 只预测最后一个位置
        prob = model.project(decoder_output[:, -1])

        # Greedy Search：选概率最大的 Token
        next_word = torch.argmax(prob, dim=1).item()

        # 拼接预测结果
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]], dtype=source.dtype, device=device)], dim=1)

        # 遇到 [EOS] 停止
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):

    model.eval()
    count = 0

    # 验证阶段不计算梯度
    with torch.no_grad():
        for batch in validation_ds:

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # 验证时 batch_size=1
            assert encoder_input.size(0) == 1

            # 生成翻译结果
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # Token IDs -> 中文文本
            model_out_text = tokenizer_tgt.decode(model_out.cpu().tolist(), skip_special_tokens=True)

            # 打印结果
            print_msg("-" * 80)
            print_msg(f"SOURCE:    {source_text}")
            print_msg(f"TARGET:    {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            count += 1
            if count >= num_examples:
                break

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer=trainer
        )

        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):

    # English -> Chinese
    ds_raw = load_dataset(
        "Helsinki-NLP/news_commentary",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train"
    )

    # 创建英文 tokenizer
    tokenizer_src = get_or_build_tokenizer(
        config,
        ds_raw,
        config["lang_src"]
    )

    # 创建中文 tokenizer
    tokenizer_tgt = get_or_build_tokenizer(
        config,
        ds_raw,
        config["lang_tgt"]
    )

    # 过滤超过 seq_len 的句子
    def valid_length(item):
        src_text = item["translation"][config["lang_src"]]
        tgt_text = item["translation"][config["lang_tgt"]]

        src_len = len(tokenizer_src.encode(src_text).ids)
        tgt_len = len(tokenizer_tgt.encode(tgt_text).ids)

        # Encoder: [SOS] + src + [EOS]
        # Decoder: [SOS] + tgt
        return (
            src_len <= config["seq_len"] - 2
            and tgt_len <= config["seq_len"] - 1
        )

    original_size = len(ds_raw)
    ds_raw = ds_raw.filter(valid_length)

    print(f"Original dataset size: {original_size}")
    print(f"Filtered dataset size: {len(ds_raw)}")
    print(f"Removed samples: {original_size - len(ds_raw)}")

    # 90% training, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(
        ds_raw,
        [train_ds_size, val_ds_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model



def train_model(config):
    #Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #TensorBoard
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    
    initial_epoch = 0
    global_step = 0

    #断点续训
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config["num_epochs"]):
        

        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            model.train()

            encoder_input = batch["encoder_input"].to(device)  # (B, Seq_Len)
            decoder_input = batch["decoder_input"].to(device)  # (B, Seq_Len)
            encoder_mask = batch["encoder_mask"].to(device)    # (B, 1, 1, Seq_Len)
            decoder_mask = batch["decoder_mask"].to(device)    # (B, 1, Seq_Len, Seq_Len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  #(B, Seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)    #(B, Seq_len, d_model)
            proj_output = model.project(decoder_output) #(B, Seq_len, tgt_vocab_size)
            label = batch['label'].to(device)   #(B, Seq_len)

            # (B, Seq_Len, tgt_vocab_size) -> (B * Seq_Len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            #Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        #Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename
        )


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)

