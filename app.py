import os
import json
import requests
import tempfile
import streamlit as st
import torch

# Import model class (assumes model.py lives next to app.py)
from model import TransformerSeq2Seq

st.set_page_config(page_title="Urdu Chat Bot", layout="centered")

st.markdown(
    """
    <div style="text-align:center; margin-bottom:12px;">
      <h1 style="color:#7B3FE4; font-family: 'Helvetica Neue', Arial;">Urdu Chat Bot</h1>
      <p style="color:#666; margin-top:-10px;">Character-level Transformer Chatbot</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Default raw URLs (uses the commit OID available in your repo snapshot)
BASE_RAW = "https://raw.githubusercontent.com/manesh230/Urdu-Conversational-Chatbot-Transformer-with-Multi-Head-Attention-/d7b7b9ecf73e2e832d3df9ddfb79a888b647d62c"
DEFAULT_CHAR2ID = f"{BASE_RAW}/char2id.json"
DEFAULT_ID2CHAR = f"{BASE_RAW}/id2char.json"
DEFAULT_TOKENIZER_CFG = f"{BASE_RAW}/tokenizer_config.json"
# default release asset (attempt)
DEFAULT_RELEASE_MODEL = "https://github.com/manesh230/Urdu-Conversational-Chatbot-Transformer-with-Multi-Head-Attention-/releases/download/v.1/model_best.pt"

st.sidebar.header("Files & Settings")
st.sidebar.markdown("If files are already present in the app folder, they will be used. Otherwise the app will attempt to download them from the provided URLs.")

char2id_url = st.sidebar.text_input("char2id.json URL", DEFAULT_CHAR2ID)
id2char_url = st.sidebar.text_input("id2char.json URL", DEFAULT_ID2CHAR)
tokenizer_cfg_url = st.sidebar.text_input("tokenizer_config.json URL", DEFAULT_TOKENIZER_CFG)
model_url = st.sidebar.text_input("model checkpoint URL", DEFAULT_RELEASE_MODEL)

max_gen_len = st.sidebar.slider("Max generation length (chars)", min_value=16, max_value=512, value=128, step=8)
use_beam = st.sidebar.checkbox("Use beam search (small beam implemented)", value=False)
beam_size = st.sidebar.slider("Beam size", min_value=2, max_value=8, value=4) if use_beam else 1

# utility functions
def download(url: str, dst: str, chunk_size=1 << 20):
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return dst


def ensure_vocab_files(char2id_path, id2char_path, cfg_path, urls):
    # urls: (char2id_url, id2char_url, tokenizer_cfg_url)
    for out, url in zip((char2id_path, id2char_path, cfg_path), urls):
        if os.path.exists(out):
            continue
        st.text(f"Downloading {os.path.basename(out)} ...")
        try:
            download(url, out)
        except Exception as e:
            st.error(f"Could not download {url}: {e}")
            raise


@st.cache_resource(show_spinner=True)
def load_vocab_and_config(char2id_path, id2char_path, cfg_path, urls):
    ensure_vocab_files(char2id_path, id2char_path, cfg_path, urls)
    with open(char2id_path, "r", encoding="utf-8") as f:
        char2id = json.load(f)
    with open(id2char_path, "r", encoding="utf-8") as f:
        id2char = json.load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        tok_conf = json.load(f)
    # ensure types
    return char2id, id2char, tok_conf


@st.cache_resource(show_spinner=True)
def load_model_and_weights(model_path, vocab_size, pad_id, try_to_download=True, download_url=None):
    if not os.path.exists(model_path):
        if try_to_download and download_url:
            st.text("Downloading model checkpoint ...")
            try:
                download(download_url, model_path)
            except Exception as e:
                st.error(f"Failed to download model checkpoint: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    # load
    ckpt = torch.load(model_path, map_location="cpu")
    # ckpt may be a dict with model_state and cfg, or a raw state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt and "cfg" in ckpt:
        cfg = ckpt["cfg"]
        d_model = cfg.get("d_model", 256)
        n_heads = cfg.get("n_heads", 2)
        enc_layers = cfg.get("enc_layers", 2)
        dec_layers = cfg.get("dec_layers", 2)
        d_ff = cfg.get("d_ff", 1024)
        dropout = cfg.get("dropout", 0.1)
        max_len = cfg.get("max_len", 256)
        model = TransformerSeq2Seq(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_id=pad_id,
            tie_embeddings=True,
            max_len=max_len,
        )
        model.load_state_dict(ckpt["model_state"])
    else:
        # raw state_dict, assume default architecture; warn user if sizes mismatch
        model = TransformerSeq2Seq(vocab_size=vocab_size, pad_id=pad_id)
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            raise RuntimeError(f"Failed to load state_dict into default model: {e}")
    model.eval()
    return model


# prepare temporary storage
DATA_DIR = os.path.join(os.getcwd(), "deployed_vocab")
os.makedirs(DATA_DIR, exist_ok=True)
char2id_path = os.path.join(DATA_DIR, "char2id.json")
id2char_path = os.path.join(DATA_DIR, "id2char.json")
cfg_path = os.path.join(DATA_DIR, "tokenizer_config.json")
model_path = os.path.join(DATA_DIR, "model_best.pt")

# Load vocab and tokenizer config (download if necessary)
try:
    char2id, id2char, tok_conf = load_vocab_and_config(char2id_path, id2char_path, cfg_path, (char2id_url, id2char_url, tokenizer_cfg_url))
except Exception:
    st.stop()

pad_id = tok_conf["special_token_ids"]["pad_id"]
sos_id = tok_conf["special_token_ids"]["sos_id"]
eos_id = tok_conf["special_token_ids"]["eos_id"]
unk_id = tok_conf["special_token_ids"]["unk_id"]
vocab_size = tok_conf["vocab_size"]

# Load model (download if necessary)
try:
    model = load_model_and_weights(model_path, vocab_size=vocab_size, pad_id=pad_id, try_to_download=True, download_url=model_url)
except Exception as e:
    st.warning(f"Model could not be loaded automatically: {e}")
    st.info("You can upload a checkpoint file below (model_best.pt or a state_dict).")
    uploaded = st.file_uploader("Upload model checkpoint (.pt or .pth)", type=["pt", "pth"])
    model = None
    if uploaded is not None:
        tmp = os.path.join(DATA_DIR, uploaded.name)
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            model = load_model_and_weights(tmp, vocab_size=vocab_size, pad_id=pad_id, try_to_download=False, download_url=None)
            st.success("Model loaded from uploaded checkpoint.")
        except Exception as e2:
            st.error(f"Upload load failed: {e2}")
            st.stop()

# optional normalization import
try:
    import preprocess_urdu_char_level as prep

    def normalize_urdu(s: str) -> str:
        return prep.normalize_urdu(s)
except Exception:
    # fallback: light normalization
    def normalize_urdu(s: str) -> str:
        return s.strip()


def tokenize_text(txt: str):
    txt = normalize_urdu(txt)
    return [char2id.get(ch, unk_id) for ch in txt]


def detokenize_ids(ids):
    out_chars = []
    for i in ids:
        if i == eos_id:
            break
        if i == pad_id:
            continue
        # id could be int or string (if id2char loaded as strings of ints) -> ensure int
        out_chars.append(id2char[int(i)])
    return "".join(out_chars)


st.markdown("---")
st.subheader("Try it out")
prefix = st.text_area("Prefix (Urdu):", value="", height=120, placeholder="Write an Urdu prefix here...")
col1, col2 = st.columns([1, 1])
with col1:
    generate_btn = st.button("Generate")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

if generate_btn:
    if model is None:
        st.error("Model not loaded. Upload checkpoint or provide valid model URL in sidebar.")
    elif not prefix or len(prefix.strip()) == 0:
        st.warning("Please enter a prefix in Urdu.")
    else:
        src_ids = tokenize_text(prefix)
        if len(src_ids) == 0:
            st.warning("Prefix tokenized to empty sequence after normalization.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

            if use_beam and beam_size > 1:
                # simple per-example beam search (small beam). Not optimized for batch.
                def beam_decode_one(model, src_tensor, sos_id, eos_id, max_len, beam):
                    model.eval()
                    mem = model.encode(src_tensor)
                    src_mask = model.make_src_key_padding_mask(src_tensor)
                    beams = [([sos_id], 0.0)]
                    finished = []
                    for _ in range(max_len - 1):
                        new_beams = []
                        for seq, score in beams:
                            if seq[-1] == eos_id:
                                finished.append((seq, score))
                                continue
                            tgt = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                            dec_out = model.decode(tgt, mem, src_mask)
                            logits = model.generator(dec_out[:, -1, :])  # (1, V)
                            logp = torch.log_softmax(logits, dim=-1).squeeze(0)
                            topk = torch.topk(logp, beam)
                            for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                                new_beams.append((seq + [int(idx)], score + float(lp)))
                        new_beams.sort(key=lambda x: x[1], reverse=True)
                        beams = new_beams[:beam]
                        if all(b[0][-1] == eos_id for b in beams):
                            finished.extend(beams)
                            break
                    if not finished:
                        finished = beams
                    finished.sort(key=lambda x: x[1], reverse=True)
                    return finished[0][0]

                seq = beam_decode_one(model, src_tensor, sos_id, eos_id, max_gen_len, beam_size)
                gen_ids = seq[1:]
                if eos_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(eos_id)]
                pred_txt = detokenize_ids(gen_ids)
            else:
                gen = model.greedy_decode(src_tensor, sos_id, eos_id, max_len=max_gen_len).cpu().tolist()
                gen_ids = gen[0][1:]
                if eos_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(eos_id)]
                pred_txt = detokenize_ids(gen_ids)

            st.markdown("### Generated suffix")
            st.success(pred_txt)

            st.markdown("### Debug / Info")
            st.write("Normalized prefix:", normalize_urdu(prefix))
            st.write("Tokens (encoder):", src_ids)
            st.write("Length (chars):", len(pred_txt))
            st.download_button("Download result (.txt)", data=pred_txt, file_name="generated_suffix.txt", mime="text/plain")

st.markdown("---")
st.markdown(
    "<div style='color:#666; font-size:13px'>Notes: The app expects the same normalization used during training. If the repo contains preprocess_urdu_char_level.py, it will be used automatically. If model download fails, upload a checkpoint file in the sidebar.</div>",
    unsafe_allow_html=True,
)
